from qdrant_ingester.config import Settings


def _set_required_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")
    monkeypatch.setenv("DOCUMENT_CHUNKER_URL", "http://chunker:8001/chunk")
    monkeypatch.setenv("QDRANT_HOST", "qdrant")


def test_allowed_collections_comma_separated(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "documents, erp_code")

    settings = Settings()

    assert settings.allowed_collections == ("documents", "erp_code")


def test_allowed_collections_json_array(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_COLLECTIONS", '["documents", "erp_code"]')

    settings = Settings()

    assert settings.allowed_collections == ("documents", "erp_code")


def test_allowed_collections_single_value(monkeypatch):
    # the docker-compose default: ALLOWED_COLLECTIONS=documents
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "documents")

    settings = Settings()

    assert settings.allowed_collections == ("documents",)


def test_allowed_collections_rejects_bad_names(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("ALLOWED_COLLECTIONS", "documents,bad name!")

    try:
        Settings()
    except Exception as exc:
        assert "Invalid collection names" in str(exc)
    else:
        raise AssertionError("expected validation error for invalid collection name")
