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

class SyncRequest(BaseModel):
    collection: str
    current_file_paths: list[str]   # absolute paths currently on disk

class SyncResponse(BaseModel):
    collection: str
    new_files: list[str]
    deleted_chunks: int