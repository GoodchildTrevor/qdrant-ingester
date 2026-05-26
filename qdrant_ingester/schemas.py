from typing import Any, List, Dict, Optional
from pydantic import BaseModel

# ---- Incoming from document-chunker ----------------------------------------

class ChunkSchema(BaseModel):
    raw: str
    lemmas: str
    meta: dict[str, Any] = {}

class ChunkResponse(BaseModel):
    file_name: str
    file_format: str
    creation_date: str
    modification_date: str
    chunks: list[ChunkSchema]

# ---- Ingest request / response ----------------------------------------------

class IngestRequest(BaseModel):
    collection: str
    file_path: str          # original path on disk (stored in Qdrant payload)
    chunk_size: int | None = None
    overlap: int | None = None
    extra_payload: dict[str, Any] = {}  # optional caller-supplied metadata merged into Qdrant payload
    inline_threshold: int | None = None  # if set: return raw text instead of ingesting when token_count <= threshold

class IngestResponse(BaseModel):
    collection: str
    file_name: str
    status: str = "success"  # "success", "partial", "failed"
    partial: bool = False
    message: Optional[str] = None
    chunks_total: int = 0
    chunks_upserted: int
    chunks_failed: int = 0
    failed_batches: List[Dict[str, Any]] = []
    token_count: Optional[int] = None   # populated when inline_threshold is provided
    inline_text: Optional[str] = None   # populated when token_count <= inline_threshold

class SyncRequest(BaseModel):
    collection: str
    current_file_paths: list[str]   # absolute paths currently on disk

class SyncResponse(BaseModel):
    collection: str
    new_files: list[str]
    deleted_chunks: int
