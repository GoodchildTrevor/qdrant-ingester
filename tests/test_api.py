import pytest
from httpx import AsyncClient

from qdrant_ingester import main
from qdrant_ingester.schemas import ChunkResponse, ChunkSchema


@pytest.fixture
def dummy_settings(monkeypatch, tmp_path):
    class DummySettings:
        document_chunker_url = "http://chunker"
        qdrant_host = "localhost"
        qdrant_port = 6333
        dense_model_name = "dense"
        sparse_model_name = "sparse"
        batch_size = 16
        upsert_batch_size = 16
        scroll_limit = 100
        chunk_size = 512
        overlap = 1
        max_file_size_mb = 100
        disable_file_size_limit = False
        api_key = "secret"
        ingest_root = tmp_path
        allowed_collections = ("documents",)
        dense_vector_config = "dense"
        sparse_vector_config = "sparse"
        debug_errors = True

    monkeypatch.setattr(main, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(main, "require_api_key", lambda api_key=None: None)
    monkeypatch.setattr(main, "require_allowed_collection", lambda collection, settings: None)
    return DummySettings()


@pytest.mark.asyncio
async def test_ingest_route_pipelines(monkeypatch, tmp_path, dummy_settings):
    target_file = tmp_path / "doc.txt"
    target_file.write_text("content")

    chunk = ChunkSchema(raw="raw", lemmas="lemmas", meta={"tokens": 10})
    chunk_resp = ChunkResponse(
        file_name="doc.txt",
        file_format="text/plain",
        creation_date="2024-01-01T00:00:00Z",
        modification_date="2024-01-01T00:00:00Z",
        chunks=[chunk],
    )

    async def fake_fetch_chunks(*args, **kwargs):
        return chunk_resp

    async def fake_embed_texts(*args, **kwargs):
        return [[0.1]], [dict(length=7)]

    async def fake_upsert_data(*args, **kwargs):
        return {
            "chunks_total": 1,
            "chunks_upserted": 1,
            "chunks_failed": 0,
            "failed_batches": [],
        }

    monkeypatch.setattr(main, "fetch_chunks", fake_fetch_chunks)
    monkeypatch.setattr(main, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(main, "upsert_data", fake_upsert_data)

    async with AsyncClient(app=main.app, base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={
                "collection": "documents",
                "file_path": str(target_file),
                "chunk_size": 512,
                "overlap": 1,
            },
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["chunks_total"] == 1


@pytest.mark.asyncio
async def test_ingest_extra_payload_reserved(monkeypatch, tmp_path, dummy_settings):
    target_file = tmp_path / "doc.txt"
    target_file.write_text("content")

    async def fake_fetch_chunks(*args, **kwargs):
        return ChunkResponse(
            file_name="doc",
            file_format="txt",
            creation_date="1",
            modification_date="1",
            chunks=[],
        )

    monkeypatch.setattr(main, "fetch_chunks", fake_fetch_chunks)
    monkeypatch.setattr(main, "embed_texts", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(main, "upsert_data", lambda *args, **kwargs: {})

    async with AsyncClient(app=main.app, base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={
                "collection": "documents",
                "file_path": str(target_file),
                "extra_payload": {"name": "bad"},
            },
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 422
