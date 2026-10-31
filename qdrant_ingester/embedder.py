import asyncio
import inspect
import logging
from typing import Any

from fastembed import TextEmbedding, SparseTextEmbedding
from more_itertools import chunked

logger = logging.getLogger(__name__)


async def _embed_batch(model: Any, batch: list[str]) -> list[Any]:
    """Run any embedder (sync fastembed or async custom) on a single batch."""
    embed = getattr(model, "embed", None)
    if embed is None:
        raise TypeError(f"{type(model).__name__} has no 'embed' method")

    if inspect.iscoroutinefunction(embed):
        result = await embed(batch)
        return list(result)

    return await asyncio.to_thread(lambda: list(embed(batch)))


async def embed_texts(
    texts: list[str],
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    batch_size: int = 16,
) -> tuple[list[list[float]], list[Any]]:
    """
    Compute dense and sparse embeddings for a list of texts.
    Sync models are run in a thread pool to avoid blocking the event loop;
    async models are awaited directly.
    Returns (dense_embeddings, sparse_embeddings).
    """
    dense_all: list[list[float]] = []
    sparse_all: list[Any] = []

    for batch in chunked(texts, batch_size):
        batch = list(batch)
        dense_batch, sparse_batch = await asyncio.gather(
            _embed_batch(dense_model, batch),
            _embed_batch(sparse_model, batch),
        )
        dense_all.extend(dense_batch)
        sparse_all.extend(sparse_batch)

    logger.info(
        "Embedded %d texts (%d batches)",
        len(texts),
        -(-len(texts) // batch_size),
    )
    return dense_all, sparse_all
