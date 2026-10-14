import json

import pytest
import httpx
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

    settings_instance = DummySettings()
    monkeypatch.setattr(main, "get_settings", lambda: settings_instance)

    main.app.dependency_overrides[main.require_api_key] = lambda: None
    main.app.dependency_overrides[main.require_allowed_collection] = lambda: None

    yield settings_instance

    main.app.dependency_overrides.clear()


ERP_JSONL = "\n".join([
    json.dumps({
        "chunk_id": "standalone_function.AAA_CALC_SUM_FACT_PAY.acf2a0959bb9",
        "package_name": None,
        "procedure_name": "AAA_CALC_SUM_FACT_PAY",
        "kind": "standalone_function",
        "content": "CREATE OR REPLACE FUNCTION AAA_CALC_SUM_FACT_PAY",
    }),
    json.dumps({
        "chunk_id": "package_procedure.BBB.acf2a0959bb9",
        "package_name": "BBB_PKG",
        "procedure_name": "DO_STUFF",
        "kind": "package_procedure",
        "content": "PROCEDURE DO_STUFF IS BEGIN NULL; END;",
    }),
])


def _patch_pipeline(monkeypatch, captured=None):
    """Stub chunker (must not be called), embedder and upsert."""

    async def must_not_call(*args, **kwargs):
        raise AssertionError("document-chunker must not be called for pre-chunked files")

    async def fake_ensure_collection(*args, **kwargs):
        return None

    async def fake_embed_texts(texts, *args, **kwargs):
        if captured is not None:
            captured["texts"] = list(texts)
        return [[0.1] for _ in texts], [dict(length=1) for _ in texts]

    async def fake_upsert_data(*args, **kwargs):
        if captured is not None:
            captured["chunks"] = kwargs["chunks"]
            captured["base_payload"] = kwargs["base_payload"]
        return {
            "chunks_total": len(kwargs["chunks"]),
            "chunks_upserted": len(kwargs["chunks"]),
            "chunks_failed": 0,
            "failed_batches": [],
        }

    monkeypatch.setattr(main, "fetch_chunks", must_not_call)
    monkeypatch.setattr(main, "_ensure_collection", fake_ensure_collection)
    monkeypatch.setattr(main, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(main, "upsert_data", fake_upsert_data)


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

    async def fake_ensure_collection(*args, **kwargs):
        return None

    monkeypatch.setattr(main, "fetch_chunks", fake_fetch_chunks)
    monkeypatch.setattr(main, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(main, "upsert_data", fake_upsert_data)
    monkeypatch.setattr(main, "_ensure_collection", fake_ensure_collection)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
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
    assert payload["prechunked"] is False


@pytest.mark.asyncio
async def test_ingest_prechunked_jsonl_skips_chunker(monkeypatch, tmp_path, dummy_settings):
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    (collection_dir / "erp.jsonl").write_text(ERP_JSONL, encoding="utf-8")

    captured = {}
    _patch_pipeline(monkeypatch, captured)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={"collection": "documents", "file_name": "erp.jsonl"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["chunks_total"] == 2
    assert payload["chunks_upserted"] == 2
    assert payload["prechunked"] is True

    # embeddings are computed over the chunk content (no lemmas field in JSONL)
    assert captured["texts"] == [
        "CREATE OR REPLACE FUNCTION AAA_CALC_SUM_FACT_PAY",
        "PROCEDURE DO_STUFF IS BEGIN NULL; END;",
    ]
    # per-chunk metadata is passed through to the upsert
    assert captured["chunks"][0]["meta"]["procedure_name"] == "AAA_CALC_SUM_FACT_PAY"
    assert captured["chunks"][0]["meta"]["kind"] == "standalone_function"
    assert captured["chunks"][1]["meta"]["package_name"] == "BBB_PKG"


@pytest.mark.asyncio
async def test_ingest_prechunked_forced_for_txt_file(monkeypatch, tmp_path, dummy_settings):
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    # .txt extension would normally go to document-chunker
    (collection_dir / "erp.txt").write_text(
        json.dumps({"content": "chunk text"}) + "\n", encoding="utf-8"
    )

    captured = {}
    _patch_pipeline(monkeypatch, captured)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={
                "collection": "documents",
                "file_name": "erp.txt",
                "prechunked": True,
            },
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    assert response.json()["prechunked"] is True
    assert captured["chunks"][0]["raw"] == "chunk text"


@pytest.mark.asyncio
async def test_ingest_prechunked_invalid_file_returns_422(monkeypatch, tmp_path, dummy_settings):
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    (collection_dir / "broken.jsonl").write_text('{"content": "ok"}\n{oops}', encoding="utf-8")

    _patch_pipeline(monkeypatch)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={"collection": "documents", "file_name": "broken.jsonl"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 422
    assert "line 2" in response.json()["detail"]


@pytest.mark.asyncio
async def test_ingest_upload_jsonl(monkeypatch, tmp_path, dummy_settings):
    captured = {}
    _patch_pipeline(monkeypatch, captured)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/ingest_upload",
            data={"collection": "documents"},
            files={"file": ("erp.jsonl", ERP_JSONL.encode("utf-8"), "application/x-ndjson")},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["file_name"] == "erp.jsonl"
    assert payload["chunks_total"] == 2
    assert payload["prechunked"] is True

    # file persisted under INGEST_ROOT/<collection>/
    saved = tmp_path / "documents" / "erp.jsonl"
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == ERP_JSONL

    # base payload carries the stored file path
    assert captured["base_payload"]["file_path"] == str(saved)


@pytest.mark.asyncio
async def test_ingest_upload_sanitizes_filename(monkeypatch, tmp_path, dummy_settings):
    captured = {}
    _patch_pipeline(monkeypatch, captured)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/ingest_upload",
            data={"collection": "documents"},
            files={"file": ("../../erp.jsonl", ERP_JSONL.encode("utf-8"))},
            headers={"X-API-Key": "secret"},
        )

    # path components are stripped: stored under the collection dir, nowhere else
    assert response.status_code == 200
    saved = tmp_path / "documents" / "erp.jsonl"
    assert saved.is_file()
    assert not (tmp_path / "erp.jsonl").exists()


@pytest.mark.asyncio
async def test_ingest_upload_bad_extra_payload_returns_422(monkeypatch, tmp_path, dummy_settings):
    _patch_pipeline(monkeypatch)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/ingest_upload",
            data={"collection": "documents", "extra_payload": "{not json}"},
            files={"file": ("erp.jsonl", ERP_JSONL.encode("utf-8"))},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_files_endpoint_empty_collection(monkeypatch, tmp_path, dummy_settings):
    # Create an empty collection directory
    collection_dir = tmp_path / "empty_collection"
    collection_dir.mkdir()

    # Update ingest_root in settings to point to our tmp_path
    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("empty_collection",)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
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

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
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
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    target_file = collection_dir / "doc.txt"
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

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
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

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
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
async def test_preview_prechunked_jsonl(monkeypatch, tmp_path, dummy_settings):
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()
    (collection_dir / "erp.jsonl").write_text(ERP_JSONL, encoding="utf-8")

    async def must_not_call(*args, **kwargs):
        raise AssertionError("document-chunker must not be called for pre-chunked files")

    monkeypatch.setattr(main, "fetch_chunks", must_not_call)

    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("documents",)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
        response = await client.post(
            "/preview",
            json={"collection": "documents", "file_name": "erp.jsonl"},
            headers={"X-API-Key": "secret"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks_total"] == 2
    assert payload["file_format"] == "prechunked/jsonl"
    assert payload["chunks"][0]["raw"] == "CREATE OR REPLACE FUNCTION AAA_CALC_SUM_FACT_PAY"


@pytest.mark.asyncio
async def test_preview_route_file_not_found(monkeypatch, tmp_path, dummy_settings):
    # Create collection directory but no file
    collection_dir = tmp_path / "documents"
    collection_dir.mkdir()

    # Update ingest_root in settings to point to our tmp_path
    dummy_settings.ingest_root = tmp_path
    dummy_settings.allowed_collections = ("documents",)

    async with AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
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
