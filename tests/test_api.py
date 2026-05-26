import pytest
from httpx import AsyncClient

from qdrant_ingester import main
from qdrant_ingester.schemas import ChunkResponse, ChunkSchema, PreviewResponse


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
    # Create file inside the collection subdirectory (documents/)
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    target_file = collection_dir / "doc.txt"
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
                "file_name": "doc.txt",
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
async def test_list_files_endpoint_empty_collection(monkeypatch, tmp_path, dummy_settings):
    # Create an empty collection directory
    collection_dir = tmp_path / "empty_collection"
    collection_dir.mkdir()
    
    # Update ingest_root in settings to point to our tmp_path
    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("empty_collection",)
    
    async with AsyncClient(app=main.app, base_url="http://test") as client:
        response = await client.get(
            "/files/empty_collection",
            headers={"X-API-Key": "secret"},
        )
    
    assert response.status_code == 200
    payload = response.json()
    assert payload["collection"] == "empty_collection"
    assert payload["files"] == []


@pytest.mark.asyncio
async def test_list_files_endpoint_nonexistent_collection(monkeypatch, tmp_path, dummy_settings):
    # Don't create the collection directory
    
    # Update ingest_root in settings to point to our tmp_path
    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("nonexistent",)
    
    async with AsyncClient(app=main.app, base_url="http://test") as client:
        response = await client.get(
            "/files/nonexistent",
            headers={"X-API-Key": "secret"},
        )
    
    assert response.status_code == 200
    payload = response.json()
    assert payload["collection"] == "nonexistent"
    assert payload["files"] == []


@pytest.mark.asyncio
async def test_ingest_extra_payload_reserved(monkeypatch, tmp_path, dummy_settings):
    target_file = (tmp_path / "documents") / "doc.txt"
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
                "file_name": "doc.txt",
                "extra_payload": {"name": "bad"},
            },
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_preview_route(monkeypatch, tmp_path, dummy_settings):
    # Create a test file
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    target_file = collection_dir / "doc.txt"
    target_file.write_text("hello world")

    # Mock the fetch_chunks function to return a predefined response
    chunk = ChunkSchema(raw="hello world", lemmas="hello world", meta={})
    chunk_resp = ChunkResponse(
        file_name="doc.txt",
        file_format="text/plain",
        creation_date="2024-01-01T00:00:00Z",
        modification_date="2024-01-01T00:00:00Z",
        chunks=[chunk],
    )

    async def fake_fetch_chunks(*args, **kwargs):
        return chunk_resp

    monkeypatch.setattr(main, "fetch_chunks", fake_fetch_chunks)

    # Update ingest_root in settings to point to our tmp_path
    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("documents",)

    async with AsyncClient(app=main.app, base_url="http://test") as client:
        response = await client.post(
            "/preview",
            json={
                "collection": "documents",
                "file_name": "doc.txt",
                "chunk_size": 512,
                "overlap": 1,
            },
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    payload = response.json()
    
    # Validate the response structure
    preview_response = PreviewResponse(**payload)
    assert preview_response.collection == "documents"
    assert preview_response.file_name == "doc.txt"
    assert preview_response.file_format == "text/plain"
    assert preview_response.chunks_total == 1
    
    # Validate the chunk preview
    assert len(preview_response.chunks) == 1
    chunk_preview = preview_response.chunks[0]
    assert chunk_preview.index == 0
    assert chunk_preview.raw == "hello world"
    assert chunk_preview.lemmas == "hello world"
    assert chunk_preview.char_count == 11
    assert chunk_preview.word_count == 2


@pytest.mark.asyncio
async def test_preview_route_file_not_found(monkeypatch, tmp_path, dummy_settings):
    # Create collection directory but no file
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()

    # Update ingest_root in settings to point to our tmp_path
    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("documents",)

    async with AsyncClient(app=main.app, base_url="http://test") as client:
        response = await client.post(
            "/preview",
            json={
                "collection": "documents",
                "file_name": "nonexistent.txt",
                "chunk_size": 512,
                "overlap": 1,
            },
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 404
