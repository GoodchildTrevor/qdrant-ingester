# qdrant-ingester

FastAPI microservice that receives document file paths, sends them to document-chunker for parsing/chunking, embeds chunk lemmas (dense + sparse) and upserts resulting points into Qdrant.

## Highlights (new)
- Partial success reporting for /ingest: API returns status ("success", "partial", "failed") and batch-level failure info.
- Idempotent upserts: deterministic point IDs derived from (file_path, chunk_index) to prevent duplicates on retries.
- Per-batch retries with exponential backoff and collection of final failed batches.
- Streaming upload to document-chunker to avoid reading whole files into memory.
- Max upload size enforcement (configurable via env): returns 413 if exceeded.
- Clear error handling: upstream errors (document-chunker) return 502; embedding or full pipeline failures return 500.

## Architecture

```
  caller
    │
    ├── POST /ingest  ──►  document-chunker /chunk  ──►  embed  ──►  Qdrant (upsert)
    └── POST /sync    ──►  diff filesystem vs Qdrant  ──►  delete orphans
```

## API

### POST /ingest

Full pipeline for a single file: send file to document-chunker → embed lemmas → upsert into Qdrant.

Request (JSON):
```json
{
  "collection": "my_collection",
  "file_path": "/data/documents/report.pdf",
  "chunk_size": 512,
  "overlap": 1
}
```
- chunk_size and overlap are optional and default to env values.

Behavior notes:
- If file size > MAX_FILE_SIZE_MB the endpoint returns 413 Payload Too Large.
- If document-chunker is unreachable or returns non-2xx, /ingest returns 502 (upstream).
- If embedding fails, /ingest returns 500 (pipeline failure).
- Upserts are idempotent: point ids are deterministic (sha1 of "file_path:chunk_index").
- Upsert is performed in batches with retries; batches that still fail after retries are reported in the response rather than aborting the whole operation.

Responses:

Success:
```json
{
  "collection": "my_collection",
  "file_name": "report.pdf",
  "status": "success",
  "chunks_total": 42,
  "chunks_upserted": 42,
  "chunks_failed": 0,
  "failed_batches": []
}
```

Partial success (some batches failed after retries):
```json
{
  "collection": "my_collection",
  "file_name": "report.pdf",
  "status": "partial",
  "chunks_total": 50,
  "chunks_upserted": 42,
  "chunks_failed": 8,
  "failed_batches": [
    {
      "batch_index": 3,
      "attempts": 3,
      "error": "HTTPError: 503 Service Unavailable",
      "size": 4,
      "ids": ["<id1>", "<id2>", "<id3>", "<id4>"]
    }
  ]
}
```

Failed (no points accepted):
```json
{
  "collection": "my_collection",
  "file_name": "report.pdf",
  "status": "failed",
  "chunks_total": 50,
  "chunks_upserted": 0,
  "chunks_failed": 50,
  "failed_batches": [ ... ]
}
```

### POST /sync

Compare a list of current file paths against what is stored in Qdrant; returns new files and deletes orphaned chunks.

Request:
```json
{
  "collection": "my_collection",
  "current_file_paths": ["/data/documents/a.pdf", "/data/documents/b.docx"]
}
```

Response:
```json
{
  "collection": "my_collection",
  "new_files": ["/data/documents/b.docx"],
  "deleted_chunks": 17
}
```

### GET /health
Returns:
```json
{"status": "ok"}
```

## Run

```bash
cp .env.example .env
# set DOCUMENT_CHUNKER_URL, QDRANT_HOST, API_KEY, INGEST_ROOT (and other env vars as needed)
docker compose up --build
```

Or run locally:
```bash
python -m uvicorn qdrant_ingester.main:app --port 8002
```

## Environment variables

`DOCUMENT_CHUNKER_URL`, `QDRANT_HOST`, and `API_KEY` are required.

| Variable | Default | Description |
|---|---|---|
| DOCUMENT_CHUNKER_URL | *(required)* | URL of the document-chunker `/chunk` endpoint |
| QDRANT_HOST | *(required)* | Qdrant hostname |
| QDRANT_PORT | 6333 | Qdrant port |
| API_KEY | *(required)* | Shared API key expected in `X-API-Key` header |
| INGEST_ROOT | /data | Base directory allowed for ingest/sync file access |
| DENSE_MODEL_NAME | sentence-transformers/paraphrase-multilingual-pmnet-base-v2 | fastembed dense model |
| SPARSE_MODEL_NAME | Qdrant/bm25 | fastembed sparse (BM25) model |
| BATCH_SIZE | 16 | Embedding batch size |
| UPSERT_BATCH_SIZE | 16 | Qdrant upsert batch size |
| SCROLL_LIMIT | 1000 | Qdrant scroll page size |
| CHUNK_SIZE | 512 | Default chunk_size forwarded to document-chunker |
| OVERLAP | 1 | Default overlap forwarded to document-chunker |
| MAX_FILE_SIZE_MB | 50 | Maximum allowed upload file size (MB) |
| APP_PORT | 8002 | Host port (docker-compose only) |

## Implementation details (notes for operators)
- Idempotency: deterministic sha1(file_path:chunk_index) point ids prevent duplicates on re-ingest.
- Retries: upsert is retried per batch with exponential backoff; final failed batches are included in the API response so callers can inspect and re-run if desired.
- Streaming: the service streams files to document-chunker (file descriptor passed to httpx) to avoid reading large files fully into memory.
- Status semantics: `partial` means some batches failed after retries; `failed` means no points were accepted. The service only returns 500 on pipeline-level failures (embedding or internal unhandled errors). Document-chunker failures are surfaced as 502 to indicate upstream issues.
