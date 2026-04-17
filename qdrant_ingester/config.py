from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding, SparseTextEmbedding
from pydantic import field as pydantic_field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings
from qdrant_client import AsyncQdrantClient


class Settings(BaseSettings):
    document_chunker_url: str = pydantic_field(
        description="URL of the document-chunker /chunk endpoint"
    )
    qdrant_host: str = pydantic_field(
        description="Qdrant hostname"
    )
    qdrant_port: int = pydantic_field(
        default=6333,
        description="Qdrant port",
    )
    dense_model_name: str = pydantic_field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        description="fastembed dense model name",
    )
    sparse_model_name: str = pydantic_field(
        default="Qdrant/bm25",
        description="fastembed sparse (BM25) model name",
    )
    batch_size: int = pydantic_field(
        default=16,
        description="Embedding batch size",
    )
    upsert_batch_size: int = pydantic_field(
        default=16,
        description="Qdrant upsert batch size",
    )
    scroll_limit: int = pydantic_field(
        default=1000,
        description="Qdrant scroll page size",
    )
    chunk_size: int = pydantic_field(
        default=512,
        description="chunk_size forwarded to document-chunker",
    )
    overlap: int = pydantic_field(
        default=1,
        description="overlap forwarded to document-chunker",
    )
    max_file_size_mb: int = pydantic_field(
        default=50,
        ge=1,
        description="Maximum allowed file size for ingest in megabytes",
    )
    disable_file_size_limit: bool = pydantic_field(
        default=False,
        description="If true, disables file size checks",
    )
    ingest_root: str = pydantic_field(
        default="/data",
        description="Base directory allowed for ingest/sync file access",
    )
    api_key: str = pydantic_field(
        default="",
        description="Shared API key required on requests",
    )

    # Security / deployment
    api_key: SecretStr | None = pydantic_field(
        default=None,
        description="Simple API key for protecting endpoints (empty disables)",
    )
    ingest_root: Path = pydantic_field(
        default=Path("/data"),
        description="Path prefix restricting ingest/sync to a specific directory",
    )
    jwt_secret: str | None = pydantic_field(
        default=None,
        description="Optional JWT secret for future OAuth2/JWT authentication",
    )

    # Debug / observability
    debug_errors: bool = pydantic_field(
        default=False,
        description="Include exception details in HTTP error responses (non-prod only)",
    )
    debug_log_file: str | None = pydantic_field(
        default=None,
        description="If set, write DEBUG-level logs to this file (console keeps INFO only)",
    )

    # Qdrant connection API key (if Qdrant is configured with an API key)
    qdrant_api_key: str | None = pydantic_field(
        default=None,
        description="Optional API key to pass to Qdrant client",
    )

    @field_validator("ingest_root")
    @classmethod
    def _validate_ingest_root(cls, v: Path) -> Path:
        v = Path(v).resolve()
        if not v.is_absolute():
            raise ValueError("ingest_root must be an absolute path")
        return v

    # Vector config names used in Qdrant collection schema
    dense_vector_config: str = "dense"
    sparse_vector_config: str = "sparse"

    allowed_collections: tuple[str, ...] = pydantic_field(
        default=("documents",),
        description="Collections allowed for ingest/sync operations",
    )

    @field_validator("allowed_collections")
    @classmethod
    def _validate_allowed_collections(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("ALLOWED_COLLECTIONS must contain at least one collection")
        import re
        bad = [c for c in v if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", c)]
        if bad:
            raise ValueError(f"Invalid collection names: {bad}")
        return v

    @model_validator(mode="after")
    def _validate_required_secrets(self):
        # Ensure the service API key is configured and non-empty.
        if not self.api_key or not self.api_key.get_secret_value().strip():
            raise ValueError("API_KEY must be set and non-empty")
        # Ensure Qdrant connection API key is configured and non-empty.
        if not self.qdrant_api_key or not str(self.qdrant_api_key).strip():
            raise ValueError("QDRANT_API_KEY must be set and non-empty")
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    s = get_settings()
    # include api_key when configured
    if getattr(s, "qdrant_api_key", None):
        return AsyncQdrantClient(host=s.qdrant_host, port=s.qdrant_port, api_key=s.qdrant_api_key)
    return AsyncQdrantClient(host=s.qdrant_host, port=s.qdrant_port)


@lru_cache
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().dense_model_name)


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=get_settings().sparse_model_name)
