import asyncio
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List
import httpx

from fastembed import SparseTextEmbedding
from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode
from qdrant_client import AsyncQdrantClient


class Settings(BaseSettings):
    document_chunker_url: str = Field(
        description="URL of the document-chunker /chunk endpoint"
    )
    qdrant_host: str = Field(
        description="Qdrant hostname"
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant port",
    )
    qdrant_https: bool = Field(
        default=False,
        description="Whether to use HTTPS for Qdrant connection (default: False for internal Docker networking)",
    )
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        description="Ollama server base URL (e.g., http://localhost:11434)",
    )
    dense_model_name: str = Field(
        default="bge-m3",
        description="Ollama dense embedding model name",
    )
    sparse_model_name: str = Field(
        default="Qdrant/bm25",
        description="fastembed sparse (BM25) model name",
    )
    batch_size: int = Field(
        default=64,
        description="Embedding batch size",
    )
    upsert_batch_size: int = Field(
        default=64,
        description="Qdrant upsert batch size",
    )
    scroll_limit: int = Field(
        default=1000,
        description="Qdrant scroll page size",
    )
    chunk_size: int = Field(
        default=512,
        description="chunk_size forwarded to document-chunker",
    )
    overlap: int = Field(
        default=1,
        description="overlap forwarded to document-chunker",
    )
    max_file_size_mb: int = Field(
        default=20000,
        ge=1,
        description="Maximum allowed file size for ingest in megabytes",
    )
    disable_file_size_limit: bool = Field(
        default=False,
        description="If true, disables file size checks",
    )

    # Security / deployment
    api_key: SecretStr | None = Field(
        default=None,
        description="Simple API key for protecting endpoints (empty disables)",
    )
    ingest_root: Path = Field(
        default=Path("/data"),
        description="Path prefix restricting ingest/sync to a specific directory",
    )
    jwt_secret: str | None = Field(
        default=None,
        description="Optional JWT secret for future OAuth2/JWT authentication",
    )

    # Debug / observability
    debug_errors: bool = Field(
        default=False,
        description="Include exception details in HTTP error responses (non-prod only)",
    )
    debug_log_file: str | None = Field(
        default=None,
        description="If set, write DEBUG-level logs to this file (console keeps INFO only)",
    )

    # Qdrant connection API key (if Qdrant is configured with an API key)
    qdrant_api_key: str | None = Field(
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

    # NoDecode: pass the raw env string to the validators below instead of
    # letting pydantic-settings json.loads() it (which fails for
    # "documents,erp_code" and even the plain default "documents").
    allowed_collections: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("documents",),
        description="Collections allowed for ingest/sync operations",
    )

    @field_validator("allowed_collections", mode="before")
    @classmethod
    def _split_allowed_collections(cls, v):
        """Accept plain env values like ALLOWED_COLLECTIONS=documents,erp_code."""
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                try:
                    v = json.loads(stripped)
                except json.JSONDecodeError:
                    v = [item.strip() for item in stripped.split(",")]
            else:
                v = [item.strip() for item in stripped.split(",") if item.strip()]
        return v

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


class OllamaDenseEmbedding:
    """Dense embeddings via Ollama's /api/embed endpoint (batch-capable)."""

    def __init__(self, base_url: str, model_name: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None


    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(60.0),
            )
        return self._client


    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        client = await self._get_client()
        last_exc = None
        for attempt in range(3):
            try:
                resp = await client.post("/api/embed", json={"model": self.model_name, "input": texts})
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 404 and attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                raise
        raise last_exc


    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    s = get_settings()
    return AsyncQdrantClient(
        host=s.qdrant_host,
        port=s.qdrant_port,
        api_key=s.qdrant_api_key or None,
        https=s.qdrant_https,
    )


@lru_cache
def get_dense_model() -> OllamaDenseEmbedding:
    if os.getenv("FASTEMBED_FAKE_MODELS") == "1":
        return _DummyDenseEmbedding()
    s = get_settings()
    return OllamaDenseEmbedding(
        base_url=s.ollama_base_url,
        model_name=s.dense_model_name,
    )


@lru_cache
def get_sparse_model() -> SparseTextEmbedding:
    if os.getenv("FASTEMBED_FAKE_MODELS") == "1":
        return _DummySparseEmbedding()
    s = get_settings()
    return SparseTextEmbedding(
        model_name=s.sparse_model_name,
        specific_model_path="/app/resources/bm25",
    )


@dataclass
class _DummyDenseEmbedding:
    dim: int = 8


    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[len(text) * 1.0] * self.dim for text in texts]


@dataclass
class _DummySparseItem:
    text_len: int

    def as_object(self) -> dict[str, int]:
        return {"length": self.text_len}


@dataclass
class _DummySparseEmbedding:
    dim: int = 8

    def embed(self, texts: list[str]) -> list[_DummySparseItem]:
        return [_DummySparseItem(len(text)) for text in texts]
