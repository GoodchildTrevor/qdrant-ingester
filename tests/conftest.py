import pytest


@pytest.fixture(autouse=True)
def enable_fake_embedding_models(monkeypatch):
    """Force dummy fastembed models in all tests to avoid heavy downloads."""
    monkeypatch.setenv("FASTEMBED_FAKE_MODELS", "1")
    yield
