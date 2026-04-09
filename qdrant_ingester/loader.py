import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from more_itertools import chunked
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


async def upsert_data(
    client: AsyncQdrantClient,
    collection_name: str,
    dense_vector_config: str,
    sparse_vector_config: str,
    dense_embeddings: list[list[float]],
    sparse_embeddings: list[Any],
    base_payload: dict[str, Any],
    chunks: list[dict[str, Any]],
    batch_size: int = 16,
) -> int:
    """
    Upsert points into Qdrant in batches.
    Returns number of upserted points.
    """
    def point_generator():
        for i, chunk in enumerate(chunks):
            vector_dict: dict[str, Any] = {
                dense_vector_config: dense_embeddings[i],
                sparse_vector_config: sparse_embeddings[i].as_object(),
            }
            meta = chunk.get("_meta") or chunk.get("meta") or {}
            payload = {
                **base_payload,
                "document": chunk["raw"],
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "table_marker": meta.get("table_marker"),
                "row_index": meta.get("row_index"),
                "chunk_tokens": meta.get("tokens"),
            }
            payload = {k: v for k, v in payload.items() if v is not None}
            yield PointStruct(id=str(uuid.uuid4()), vector=vector_dict, payload=payload)

    upserted = 0
    for batch in chunked(point_generator(), batch_size):
        try:
            await client.upsert(
                collection_name=collection_name,
                points=list(batch),
                wait=True,
            )
            upserted += len(batch)
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error("Upsert batch failed: %s", e)
    logger.info("Upserted %d points into '%s'", upserted, collection_name)
    return upserted


async def sync_file_paths(
    client: AsyncQdrantClient,
    collection: str,
    current_file_paths: set[Path],
    payload_key: str = "file_path",
    scroll_limit: int = 1000,
) -> tuple[set[Path], set[str]]:
    """
    Compare current filesystem paths against what is stored in Qdrant.
    Returns (new_paths, deleted_paths).
    """
    current_str = {str(p) for p in current_file_paths}
    db_str: set[str] = set()
    offset = None
    try:
        while True:
            points, offset = await client.scroll(
                collection_name=collection,
                with_payload=[payload_key],
                with_vectors=False,
                limit=scroll_limit,
                offset=offset,
            )
            if not points:
                break
            for pt in points:
                path = pt.payload.get(payload_key)
                if path:
                    db_str.add(path)
            if not offset:
                break
    except UnexpectedResponse as e:
        if getattr(e, "response", None) and e.response.status_code == 404:
            logger.warning("Collection '%s' not found, treating as empty.", collection)
            return current_file_paths, set()
        raise

    new_paths = {Path(p) for p in (current_str - db_str)}
    deleted = current_str - db_str  # re-used as deleted = db_str - current_str
    deleted = db_str - current_str
    logger.info("%d new, %d deleted for collection '%s'", len(new_paths), len(deleted), collection)
    return new_paths, deleted


async def delete_orphaned_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    deleted_file_paths: set[str],
    payload_key: str = "file_path",
    scroll_limit: int = 1000,
) -> int:
    """Delete all points whose file_path is in deleted_file_paths."""
    if not deleted_file_paths:
        return 0
    point_ids = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection_name,
            with_payload=[payload_key],
            with_vectors=False,
            limit=scroll_limit,
            offset=offset,
        )
        if not points:
            break
        for pt in points:
            if pt.payload.get(payload_key) in deleted_file_paths:
                point_ids.append(pt.id)
        if not offset:
            break

    if point_ids:
        for batch in chunked(point_ids, 1000):
            await client.delete(
                collection_name=collection_name,
                points_selector=list(batch),
                wait=True,
            )
        logger.info("Deleted %d orphaned chunks", len(point_ids))
    return len(point_ids)
