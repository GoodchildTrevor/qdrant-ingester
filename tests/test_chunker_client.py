import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from qdrant_ingester.chunker_client import fetch_chunks
from qdrant_ingester.schemas import ChunkResponse
from tests.conftest import VALID_CHUNK_RESPONSE_JSON


class DummyResponse:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aread(self):
        return self._body


class DummyClient:
    def __init__(self, response: DummyResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @asynccontextmanager
    async def stream(self, *args, **kwargs):
        yield self._response


def make_client(status_code: int, body: bytes) -> DummyClient:
    return DummyClient(DummyResponse(status_code, body))


@pytest.mark.asyncio
async def test_fetch_chunks_success(monkeypatch, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    valid_body = json.dumps(VALID_CHUNK_RESPONSE_JSON).encode()
    client = make_client(200, valid_body)
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *a, **kw: client)

    result = await fetch_chunks(
        chunker_url="http://chunker",
        file_path=path,
        chunk_size=16,
        overlap=1,
    )

    assert isinstance(result, ChunkResponse)
    assert result.file_name == "file.txt"
    assert len(result.chunks) == 1
    assert result.chunks[0].raw == "hello world"


@pytest.mark.asyncio
async def test_fetch_chunks_http_error(monkeypatch, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    client = make_client(500, b"Internal Server Error")
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *a, **kw: client)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await fetch_chunks(
            chunker_url="http://chunker",
            file_path=path,
            chunk_size=16,
            overlap=1,
        )


@pytest.mark.asyncio
async def test_fetch_chunks_invalid_json(monkeypatch, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    client = make_client(200, b"not-json")
    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *a, **kw: client)

    with pytest.raises(Exception):
        await fetch_chunks(
            chunker_url="http://chunker",
            file_path=path,
            chunk_size=16,
            overlap=1,
        )
