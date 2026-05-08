import pytest

from qdrant_ingester.embedder import embed_texts


class DummyModel:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[len(text) * 1.0] for text in texts]


@pytest.mark.asyncio
async def test_embed_texts_batches_correctly():
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
    assert dense_embs[0] == [5.0]   # len("alpha") == 5
    assert sparse_embs[1] == [4.0]  # len("beta") == 4
    assert dense.calls[0] == ["alpha", "beta"]
    assert dense.calls[1] == ["gamma"]


@pytest.mark.asyncio
async def test_embed_texts_empty_input():
    dense = DummyModel()
    sparse = DummyModel()

    dense_embs, sparse_embs = await embed_texts(
        texts=[],
        dense_model=dense,
        sparse_model=sparse,
    )

    assert dense_embs == []
    assert sparse_embs == []
    assert dense.calls == []


@pytest.mark.asyncio
async def test_embed_texts_single_item():
    dense = DummyModel()
    sparse = DummyModel()

    dense_embs, sparse_embs = await embed_texts(
        texts=["hello"],
        dense_model=dense,
        sparse_model=sparse,
        batch_size=10,
    )

    assert len(dense_embs) == 1
    assert dense_embs[0] == [5.0]
    assert len(dense.calls) == 1
