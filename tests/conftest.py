import pytest


@pytest.fixture(autouse=True)
def enable_fake_embedding_models(monkeypatch):
    """Force dummy fastembed models in all tests to avoid heavy downloads."""
    monkeypatch.setenv("FASTEMBED_FAKE_MODELS", "1")
    yield


VALID_CHUNK_RESPONSE_JSON = {
    "file_name": "file.txt",
    "file_format": "txt",
    "creation_date": "2026-01-01",
    "modification_date": "2026-01-01",
    "chunks": [
        {"raw": "hello world", "lemmas": "hello world", "meta": {}}
    ],
}
