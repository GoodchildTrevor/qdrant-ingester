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

# ---- Preview request / response --------------------------------------------

class PreviewRequest(BaseModel):
    collection: str
    file_name: str
    chunk_size: int | None = None
    overlap: int | None = None
    prechunked: bool | None = None  # force pre-chunked mode; auto-detected by .json/.jsonl extension when None

class ChunkPreview(BaseModel):
    index: int
    raw: str
    lemmas: str
    char_count: int
    word_count: int

class PreviewResponse(BaseModel):
    collection: str
    file_name: str
    file_format: str
    chunks_total: int
    chunks: list[ChunkPreview]

# ---- Ingest request / response ----------------------------------------------

class IngestRequest(BaseModel):
    collection: str
    file_name: str
    chunk_size: int | None = None
    overlap: int | None = None
    extra_payload: dict[str, Any] = {}  # optional caller-supplied metadata merged into Qdrant payload
    inline_threshold: int | None = None  # if set: return raw text instead of ingesting when token_count <= threshold
    prechunked: bool | None = None  # file already contains chunks: parse locally, skip document-chunker (auto-detected by extension when None)

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
    prechunked: bool = False            # true when the chunker was skipped (pre-chunked JSON/JSONL input)

class SyncRequest(BaseModel):
    collection: str
    current_file_paths: list[str]   # absolute paths currently on disk

class SyncResponse(BaseModel):
    collection: str
    new_files: list[str]
    deleted_chunks: int
    
