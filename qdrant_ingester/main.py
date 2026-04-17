import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader

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

# Simple API-key protection: if Settings.api_key is empty, auth is disabled.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def require_api_key(key: str = Security(api_key_header)):
    s = get_settings()
    if s.api_key and key != s.api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    _ : None = Depends(require_api_key),
):
    """
    Full pipeline for a single file: chunk -> embed -> upsert.
    Returns partial success metadata if some upsert batches ultimately fail.
    """
    settings = get_settings()
    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    chunk_size = request.chunk_size or settings.chunk_size
    overlap = request.overlap or settings.overlap
    # Restrict ingest to an allowed root if configured to prevent LFI/exfiltration
    if getattr(settings, "ingest_root", ""):
        allowed_root = Path(settings.ingest_root).resolve()
        file_path = (allowed_root / request.file_path).resolve()
        if allowed_root not in file_path.parents and file_path != allowed_root:
            raise HTTPException(status_code=403, detail="path not allowed")
    else:
        file_path = Path(request.file_path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Enforce maximum file size
    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = None
    if size_mb is not None and size_mb > getattr(settings, "max_file_size_mb", 0):
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB)")

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
        # treat chunker as upstream; return 502 so caller can retry
        raise HTTPException(status_code=502, detail=f"document-chunker error: {e}")

    if not chunk_resp.chunks:
        logger.warning("No chunks returned for %s, skipping upsert.", file_path)
        return IngestResponse(
            collection=request.collection,
            file_name=chunk_resp.file_name,
            status="success",
            partial=False,
            message=None,
            chunks_total=0,
            chunks_upserted=0,
            chunks_failed=0,
            failed_batches=[],
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
        # Embedding considered a pipeline-level failure
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # Step 3 — upsert (idempotent, with retries). upsert_data now returns metrics dict.
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
        metrics = await upsert_data(
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
        # Treat as pipeline failure only if upsert raised unexpectedly
        raise HTTPException(status_code=500, detail=f"Qdrant upsert error: {e}")

    # Build response: partial if any failed batches remain
    status_str = "success"
    if metrics.get("chunks_failed", 0) > 0:
        status_str = "partial"
    if metrics.get("chunks_upserted", 0) == 0 and metrics.get("chunks_failed", 0) > 0:
        status_str = "failed"

    failed = metrics.get("chunks_failed", 0)
    failed_batches = metrics.get("failed_batches", [])
    partial_flag = failed > 0 and metrics.get("chunks_upserted", 0) > 0
    message = None
    if failed:
        message = f"{failed} chunks failed in {len(failed_batches)} failed batch(es)"
    return IngestResponse(
        collection=request.collection,
        file_name=chunk_resp.file_name,
        status=status_str,
        partial=partial_flag,
        message=message,
        chunks_total=metrics.get("chunks_total", 0),
        chunks_upserted=metrics.get("chunks_upserted", 0),
        chunks_failed=failed,
        failed_batches=failed_batches,
    )


@app.post("/sync", response_model=SyncResponse)
async def sync(
    request: SyncRequest,
    _ : None = Depends(require_api_key),
):
    """
    Synchronize Qdrant collection against current filesystem state:
    - Detect new files (not yet in Qdrant)
    - Delete orphaned chunks (files that no longer exist)
    Returns new file paths and count of deleted chunks.
    """
    settings = get_settings()
    client = get_qdrant_client()

    # Do not trust empty client-supplied lists. If empty and an ingest_root is configured,
    # derive current paths from the server filesystem. Otherwise require the client to supply.
    if request.current_file_paths:
        current_paths = {Path(p) for p in request.current_file_paths}
    else:
        if getattr(settings, "ingest_root", ""):
            allowed_root = Path(settings.ingest_root).resolve()
            current_paths = {p for p in allowed_root.rglob("*") if p.is_file()}
        else:
            raise HTTPException(status_code=400, detail="current_file_paths required for delete")

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
