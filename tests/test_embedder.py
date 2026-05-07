import asyncio

import pytest

from qdrant_ingester.embedder import embed_texts


@pytest.mark.asyncio
async def test_embed_texts_batches_densely_chunks(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [[len(text) * 1.0] for text in texts]

    dense = DummyModel()
    sparse = DummyModel()
    texts = ["alpha", "beta", "gamma"]
    dense_embs, sparse_embs = await embed_texts(
        texts=texts,
        dense_model=dense,
        sparse_model=sparse,
        batch_size=2,
    )

    assert len(dense_embs) == 3
    assert dense_embs[0] == [5.0]
    assert sparse_embs[1] == [4.0]
    # ensure batching happened
    assert dense.calls[0] == ["alpha", "beta"]
    assert dense.calls[1] == ["gamma"]
