import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from qdrant_ingester.config import (
    get_settings,
    get_qdrant_client,
    get_dense_model,
    get_sparse_model,
)
from qdrant_ingester.schemas import (
    IngestRequest, IngestResponse,
    SyncRequest, SyncResponse,
)
from qdrant_ingester.chunker_client import fetch_chunks
from qdrant_ingester.embedder import embed_texts
from qdrant_ingester.loader import upsert_data, sync_file_paths, delete_orphaned_chunks

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="qdrant-ingester", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
):
    """
    Full pipeline for a single file:
    1. Send file to document-chunker -> get chunks
    2. Embed chunk lemmas (dense + sparse)
    3. Upsert points into Qdrant
    """
    settings = get_settings()
    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    chunk_size = request.chunk_size or settings.chunk_size
    overlap = request.overlap or settings.overlap
    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Step 1 — chunk
    try:
        chunk_resp = await fetch_chunks(
            chunker_url=settings.document_chunker_url,
            file_path=file_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as e:
        logger.error("Chunker call failed for %s: %s", file_path, e)
        raise HTTPException(status_code=502, detail=f"document-chunker error: {e}")

    if not chunk_resp.chunks:
        logger.warning("No chunks returned for %s, skipping upsert.", file_path)
        return IngestResponse(
            collection=request.collection,
            file_name=chunk_resp.file_name,
            chunks_upserted=0,
        )

    # Step 2 — embed
    lemmas = [ch.lemmas for ch in chunk_resp.chunks]
    try:
        dense_embs, sparse_embs = await embed_texts(
            texts=lemmas,
            dense_model=dense_model,
            sparse_model=sparse_model,
            batch_size=settings.batch_size,
        )
    except Exception as e:
        logger.error("Embedding failed for %s: %s", file_path, e)
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # Step 3 — upsert
    chunks_as_dicts = [
        {"raw": ch.raw, "lemmas": ch.lemmas, "meta": ch.meta}
        for ch in chunk_resp.chunks
    ]
    base_payload = {
        "name": chunk_resp.file_name,
        "file_path": str(file_path),
        "file_format": chunk_resp.file_format,
        "creation_date": chunk_resp.creation_date,
        "modification_date": chunk_resp.modification_date,
    }
    try:
        upserted = await upsert_data(
            client=client,
            collection_name=request.collection,
            dense_vector_config=settings.dense_vector_config,
            sparse_vector_config=settings.sparse_vector_config,
            dense_embeddings=dense_embs,
            sparse_embeddings=sparse_embs,
            base_payload=base_payload,
            chunks=chunks_as_dicts,
            batch_size=settings.upsert_batch_size,
        )
    except Exception as e:
        logger.error("Upsert failed for %s: %s", file_path, e)
        raise HTTPException(status_code=500, detail=f"Qdrant upsert error: {e}")

    return IngestResponse(
        collection=request.collection,
        file_name=chunk_resp.file_name,
        chunks_upserted=upserted,
    )


@app.post("/sync", response_model=SyncResponse)
async def sync(
    request: SyncRequest,
):
    """
    Synchronize Qdrant collection against current filesystem state:
    - Detect new files (not yet in Qdrant)
    - Delete orphaned chunks (files that no longer exist)
    Returns new file paths and count of deleted chunks.
    """
    settings = get_settings()
    client = get_qdrant_client()

    current_paths = {Path(p) for p in request.current_file_paths}

    try:
        new_paths, deleted_paths = await sync_file_paths(
            client=client,
            collection=request.collection,
            current_file_paths=current_paths,
            scroll_limit=settings.scroll_limit,
        )
    except Exception as e:
        logger.error("Sync failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Sync error: {e}")

    deleted_count = 0
    if deleted_paths:
        try:
            deleted_count = await delete_orphaned_chunks(
                client=client,
                collection_name=request.collection,
                deleted_file_paths=deleted_paths,
                scroll_limit=settings.scroll_limit,
            )
        except Exception as e:
            logger.error("Orphan cleanup failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Cleanup error: {e}")

    return SyncResponse(
        collection=request.collection,
        new_files=[str(p) for p in new_paths],
        deleted_chunks=deleted_count,
    )
