from functools import lru_cache

from fastembed import TextEmbedding, SparseTextEmbedding
from pydantic import field as pydantic_field
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
        default=0,
        description="Maximum allowed file size for ingest in megabytes (0 disables check)",
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
    api_key: str = pydantic_field(
        default="",
        description="Simple API key for protecting endpoints (empty disables)",
    )
    ingest_root: str = pydantic_field(
        default="",
        description="Optional path prefix restricting ingest/sync to a specific directory (absolute or relative to service)",
    )
    jwt_secret: str | None = pydantic_field(
        default=None,
        description="Optional JWT secret for future OAuth2/JWT authentication",
    )

    # Vector config names used in Qdrant collection schema
    dense_vector_config: str = "dense"
    sparse_vector_config: str = "sparse"

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
    return AsyncQdrantClient(host=s.qdrant_host, port=s.qdrant_port)


@lru_cache
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().dense_model_name)


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=get_settings().sparse_model_name)
