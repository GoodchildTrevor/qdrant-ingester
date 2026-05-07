from pathlib import Path

import pytest

from qdrant_ingester.chunker_client import fetch_chunks


class DummyResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self._content = content

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError("status")

    async def aread(self):
        return self._content


class DummyClient:
    def __init__(self, response: DummyResponse):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def stream(self, *args, **kwargs):
        return self.response


@pytest.mark.asyncio
async def test_fetch_chunks_success(monkeypatch, tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("dummy")

    response = DummyResponse(200, b"{}")
    client = DummyClient(response)

    async def dummy_async_client(*args, **kwargs):
        return client

    monkeypatch.setattr("qdrant_ingester.chunker_client.httpx.AsyncClient", lambda *args, **kwargs: client)

    result = await fetch_chunks(
        chunker_url="http://chunker",
        file_path=path,
        chunk_size=16,
        overlap=1,
    )

    assert result is not None
