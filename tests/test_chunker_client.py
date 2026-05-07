from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qdrant_ingester.chunker_client import fetch_chunks
from qdrant_ingester.schemas import ChunkResponse, ChunkSchema
from qdrant_ingester.embedder import embed_texts


class DummyResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aread(self):
        return self._content


class DummyClient:
    """Mock httpx.AsyncClient that returns a predefined response via stream()."""
    
    def __init__(self, response: DummyResponse):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @asynccontextmanager
    async def stream(self, *args, **kwargs):
        """Async context manager that yields the response."""
        yield self.response


def make_chunk_response_json() -> bytes:
    """Return valid JSON for ChunkResponse."""
    return b'''{
        "file_name": "test.txt",
        "file_format": "txt",
        "creation_date": "2024-01-01",
        "modification_date": "2024-01-02",
        "chunks": [
            {"raw": "chunk1", "lemmas": "chunk1", "meta": {}}
        ]
    }'''


# =============================================================================
# Tests for fetch_chunks
# =============================================================================

@pytest.mark.asyncio
async def test_fetch_chunks_success(monkeypatch, tmp_path):
    """Test successful fetch_chunks returns valid ChunkResponse."""
    path = tmp_path / "file.txt"
    path.write_text("dummy content")

    response = DummyResponse(200, make_chunk_response_json())
    client = DummyClient(response)
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *args, **kwargs: client)

    result = await fetch_chunks(
        chunker_url="http://chunker",
        file_path=path,
        chunk_size=16,
        overlap=1,
    )

    assert result is not None
    assert isinstance(result, ChunkResponse)
    assert result.file_name == "test.txt"
    assert result.file_format == "txt"
    assert len(result.chunks) == 1
    assert isinstance(result.chunks[0], ChunkSchema)
    assert result.chunks[0].raw == "chunk1"


@pytest.mark.asyncio
async def test_fetch_chunks_http_4xx_error(monkeypatch, tmp_path):
    """Test fetch_chunks raises on HTTP 4xx error."""
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    response = DummyResponse(400, b'{"error": "Bad Request"}')
    client = DummyClient(response)
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        await fetch_chunks(
            chunker_url="http://chunker",
            file_path=path,
            chunk_size=16,
            overlap=1,
        )


@pytest.mark.asyncio
async def test_fetch_chunks_http_5xx_error(monkeypatch, tmp_path):
    """Test fetch_chunks raises on HTTP 5xx error."""
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    response = DummyResponse(500, b'{"error": "Internal Server Error"}')
    client = DummyClient(response)
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *args, **kwargs: client)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await fetch_chunks(
            chunker_url="http://chunker",
            file_path=path,
            chunk_size=16,
            overlap=1,
        )


@pytest.mark.asyncio
async def test_fetch_chunks_invalid_json(monkeypatch, tmp_path):
    """Test fetch_chunks raises ValueError on invalid JSON response."""
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    response = DummyResponse(200, b"not-json")
    client = DummyClient(response)
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *args, **kwargs: client)

    with pytest.raises(ValueError):
        await fetch_chunks(
            chunker_url="http://chunker",
            file_path=path,
            chunk_size=16,
            overlap=1,
        )


@pytest.mark.asyncio
async def test_fetch_chunks_nonexistent_file(monkeypatch, tmp_path):
    """Test fetch_chunks raises FileNotFoundError for non-existent file."""
    nonexistent_path = tmp_path / "nonexistent.txt"
    # Don't create the file

    response = DummyResponse(200, make_chunk_response_json())
    client = DummyClient(response)
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *args, **kwargs: client)

    # The function opens the file, so it should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        await fetch_chunks(
            chunker_url="http://chunker",
            file_path=nonexistent_path,
            chunk_size=16,
            overlap=1,
        )


# =============================================================================
# Tests for embed_texts
# =============================================================================

@pytest.mark.asyncio
async def test_embed_texts_empty_list():
    """Test embed_texts with empty list returns empty results."""
    dense_model = MagicMock()
    sparse_model = MagicMock()

    dense_embeddings, sparse_embeddings = await embed_texts(
        texts=[],
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=16,
    )

    assert dense_embeddings == []
    assert sparse_embeddings == []


@pytest.mark.asyncio
async def test_embed_texts_batch_size_zero():
    """Test embed_texts with batch_size=0 handles gracefully (no batches)."""
    dense_model = MagicMock()
    sparse_model = MagicMock()

    # With batch_size=0, chunked returns empty iterator, so result should be empty
    dense_embeddings, sparse_embeddings = await embed_texts(
        texts=["text1", "text2"],
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=0,
    )

    # When batch_size=0, more_itertools.chunked yields nothing, so results are empty
    assert dense_embeddings == []
    assert sparse_embeddings == []


@pytest.mark.asyncio
async def test_embed_texts_single_batch():
    """Test embed_texts processes texts in a single batch."""
    mock_dense_result = [[0.1, 0.2], [0.3, 0.4]]
    mock_sparse_result = [{"indices": [1, 2], "values": [0.5, 0.6]}, {"indices": [3, 4], "values": [0.7, 0.8]}]

    dense_model = MagicMock()
    dense_model.embed.return_value = iter(mock_dense_result)

    sparse_model = MagicMock()
    sparse_model.embed.return_value = iter(mock_sparse_result)

    dense_embeddings, sparse_embeddings = await embed_texts(
        texts=["text1", "text2"],
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=16,
    )

    assert dense_embeddings == mock_dense_result
    assert sparse_embeddings == mock_sparse_result


@pytest.mark.asyncio
async def test_embed_texts_multiple_batches():
    """Test embed_texts processes texts across multiple batches."""
    mock_dense_result_batch1 = [[0.1, 0.2]]
    mock_dense_result_batch2 = [[0.3, 0.4]]
    mock_sparse_result_batch1 = [{"indices": [1], "values": [0.5]}]
    mock_sparse_result_batch2 = [{"indices": [2], "values": [0.6]}]

    dense_model = MagicMock()
    dense_model.embed.side_effect = [
        iter(mock_dense_result_batch1),
        iter(mock_dense_result_batch2),
    ]

    sparse_model = MagicMock()
    sparse_model.embed.side_effect = [
        iter(mock_sparse_result_batch1),
        iter(mock_sparse_result_batch2),
    ]

    dense_embeddings, sparse_embeddings = await embed_texts(
        texts=["text1", "text2"],
        dense_model=dense_model,
        sparse_model=sparse_model,
        batch_size=1,  # Force 2 batches
    )

    assert dense_embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert sparse_embeddings == [{"indices": [1], "values": [0.5]}, {"indices": [2], "values": [0.6]}]


# =============================================================================
# Integration tests for FastAPI routes
# =============================================================================

@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.api_key.get_secret_value.return_value = "test-api-key"
    settings.qdrant_host = "localhost"
    settings.qdrant_port = 6333
    settings.qdrant_api_key = "qdrant-key"
    settings.dense_model_name = "test-dense"
    settings.sparse_model_name = "test-sparse"
    settings.chunk_size = 512
    settings.overlap = 1
    settings.allowed_collections = ["test-collection"]
    settings.ingest_root = "/tmp"
    settings.debug_errors = False
    settings.max_file_size_mb = 100
    settings.disable_file_size_limit = False
    return settings


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client."""
    client = MagicMock()
    client.upsert.return_value = None
    return client


@pytest.fixture
def mock_dense_model():
    """Create mock dense embedding model."""
    model = MagicMock()
    model.embed.return_value = iter([[0.1, 0.2, 0.3]])
    return model


@pytest.fixture
def mock_sparse_model():
    """Create mock sparse embedding model."""
    model = MagicMock()
    model.embed.return_value = iter([{"indices": [1, 2], "values": [0.5, 0.6]}])
    return model


@pytest.mark.asyncio
async def test_ingest_endpoint_success(tmp_path, mock_settings, mock_qdrant_client, mock_dense_model, mock_sparse_model):
    """Test /ingest endpoint with mocked dependencies."""
    from qdrant_ingester.main import app
    from httpx import AsyncClient, ASGITransport
    from unittest.mock import patch

    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    # Update ingest_root to tmp_path
    mock_settings.ingest_root = str(tmp_path)

    # Mock chunk response
    chunk_response_json = make_chunk_response_json()

    with patch("qdrant_ingester.main.get_settings", return_value=mock_settings), \
         patch("qdrant_ingester.main.get_qdrant_client", return_value=mock_qdrant_client), \
         patch("qdrant_ingester.main.get_dense_model", return_value=mock_dense_model), \
         patch("qdrant_ingester.main.get_sparse_model", return_value=mock_sparse_model), \
         patch("qdrant_ingester.chunker_client.httpx.AsyncClient") as mock_client_class:
        
        # Setup mock client for fetch_chunks
        mock_response = DummyResponse(200, chunk_response_json)
        mock_client = DummyClient(mock_response)
        mock_client_class.return_value = mock_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                json={
                    "collection": "test-collection",
                    "file_path": "test.txt",
                },
                headers={"X-API-Key": "test-api-key"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["collection"] == "test-collection"
    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_ingest_endpoint_unauthorized():
    """Test /ingest endpoint rejects requests without valid API key."""
    from qdrant_ingester.main import app
    from httpx import AsyncClient, ASGITransport
    from unittest.mock import patch, MagicMock

    mock_settings = MagicMock()
    mock_settings.api_key.get_secret_value.return_value = "secret-key"
    mock_settings.allowed_collections = ["test-collection"]

    with patch("qdrant_ingester.main.get_settings", return_value=mock_settings):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                json={
                    "collection": "test-collection",
                    "file_path": "/tmp/test.txt",
                },
                headers={"X-API-Key": "wrong-key"},
            )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test /health endpoint returns ok status."""
    from qdrant_ingester.main import app
    from httpx import AsyncClient, ASGITransport

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
