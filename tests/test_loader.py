import pytest
from qdrant_client.models import FilterSelector

from qdrant_ingester.loader import _make_point_id, upsert_data


class _SparseStub:
    def as_object(self):
        return {"indices": [0], "values": [1.0]}


class FakeQdrantClient:
    """Records upserts and deletes; fails the upsert batches listed in *fail_batches*."""

    def __init__(self, fail_batches=()):
        self.fail_batches = set(fail_batches)
        self.upsert_calls = 0
        self.deletes = []

    async def upsert(self, collection_name, points, wait):
        batch_index = self.upsert_calls
        self.upsert_calls += 1
        if batch_index in self.fail_batches:
            raise RuntimeError(f"batch {batch_index} rejected")

    async def delete(self, collection_name, points_selector, wait):
        self.deletes.append(points_selector)


FILE_PATH = "/data/documents/doc.txt"
BASE_PAYLOAD = {"name": "doc.txt", "file_path": FILE_PATH}
CHUNKS = [
    {"raw": "one", "lemmas": "one", "meta": {"chunk_id": "rev2-a"}},
    {"raw": "two", "lemmas": "two", "meta": {"chunk_id": "rev2-b"}},
]


async def _run(client):
    return await upsert_data(
        client=client,
        collection_name="documents",
        dense_vector_config="dense",
        sparse_vector_config="sparse",
        dense_embeddings=[[0.1], [0.2]],
        sparse_embeddings=[_SparseStub(), _SparseStub()],
        base_payload=BASE_PAYLOAD,
        chunks=CHUNKS,
        batch_size=1,
        max_attempts=1,
        initial_backoff=0.0,
    )


@pytest.mark.asyncio
async def test_stale_points_removed_after_full_success():
    client = FakeQdrantClient()

    metrics = await _run(client)

    assert metrics["chunks_failed"] == 0
    assert len(client.deletes) == 1
    selector = client.deletes[0]
    assert isinstance(selector, FilterSelector)
    assert selector.filter.must[0].match.value == FILE_PATH
    assert selector.filter.must_not[0].has_id == [
        _make_point_id(FILE_PATH, "rev2-a"),
        _make_point_id(FILE_PATH, "rev2-b"),
    ]


@pytest.mark.asyncio
async def test_stale_points_kept_after_partial_failure():
    # A failed batch means the new version is incomplete: the old points must
    # stay so a retry can finish the ingest without leaving holes.
    client = FakeQdrantClient(fail_batches={1})

    metrics = await _run(client)

    assert metrics["chunks_upserted"] == 1
    assert metrics["chunks_failed"] == 1
    assert client.deletes == []
