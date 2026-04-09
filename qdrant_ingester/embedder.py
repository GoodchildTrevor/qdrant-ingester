import asyncio
import logging
from typing import Any

from fastembed import TextEmbedding, SparseTextEmbedding
from more_itertools import chunked

logger = logging.getLogger(__name__)


async def embed_texts(
    texts: list[str],
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    batch_size: int = 16,
) -> tuple[list[list[float]], list[Any]]:
    """
    Compute dense and sparse embeddings for a list of texts.
    Models are run in a thread pool to avoid blocking the event loop.
    Returns (dense_embeddings, sparse_embeddings).
    """
    dense_all: list[list[float]] = []
    sparse_all: list[Any] = []

    for batch in chunked(texts, batch_size):
        batch = list(batch)
        dense_batch, sparse_batch = await asyncio.gather(
            asyncio.to_thread(lambda b=batch: list(dense_model.embed(b))),
            asyncio.to_thread(lambda b=batch: list(sparse_model.embed(b))),
        )
        dense_all.extend(dense_batch)
        sparse_all.extend(sparse_batch)

    logger.info("Embedded %d texts (%d batches)", len(texts), -(-len(texts) // batch_size))
    return dense_all, sparse_all
