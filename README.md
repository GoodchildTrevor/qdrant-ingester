# qdrant-ingester

FastAPI microservice that ingests documents into Qdrant via a **chunk → embed → upsert** pipeline.
Supports an optional **inline mode**: if a document is small enough (below a configurable token threshold), the service skips Qdrant entirely and returns the raw text directly to the caller.

## Quick start

1. Copy example env:
```bash
cp .env.example .env
```

2. Fill required variables (see [Environment](#environment)).

3. Run with Docker Compose (recommended):
```bash
docker compose up --build
```

Or locally:
```bash
python -m uvicorn qdrant_ingester.main:app --host 0.0.0.0 --port 8002
```

---

## API

### `POST /ingest`

Full ingestion pipeline with optional inline mode.

**Request:**
```json
{
  "collection": "documents",
  "file_path": "reports/report.pdf",
  "chunk_size": 512,
  "overlap": 50,
  "extra_payload": {"user_id": "abc123"},
  "inline_threshold": 500
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `collection` | string | ✅ | Target Qdrant collection (must be in `ALLOWED_COLLECTIONS`) |
| `file_path` | string | ✅ | Path relative to `INGEST_ROOT` (or absolute if `INGEST_ROOT` is not set) |
| `chunk_size` | int | — | Override default chunk size |
| `overlap` | int | — | Override default overlap |
| `extra_payload` | object | — | Extra metadata merged into each Qdrant point payload |
| `inline_threshold` | int | — | Token threshold for inline mode (see below) |

**Behavior — inline mode:**

If `inline_threshold` is provided:
- The service counts tokens in the document (whitespace-split approximation).
- If `token_count <= inline_threshold` → **Qdrant is not touched**. The full raw text is returned in `inline_text`. `chunks_upserted` will be `0`.
- If `token_count > inline_threshold` → the normal pipeline runs (chunk → embed → upsert), `inline_text` is `null`.

If `inline_threshold` is **not provided** → always runs the full pipeline (backward-compatible default).

**Response:**
```json
{
  "collection": "documents",
  "file_name": "report.pdf",
  "status": "success",
  "partial": false,
  "message": null,
  "chunks_total": 4,
  "chunks_upserted": 0,
  "chunks_failed": 0,
  "failed_batches": [],
  "token_count": 312,
  "inline_text": "Full document text here..."
}
```

| Field | Description |
|---|---|
| `status` | `"success"` \| `"partial"` \| `"failed"` |
| `partial` | `true` when some chunks failed but others succeeded |
| `chunks_upserted` | `0` when inline mode was triggered |
| `token_count` | Populated when `inline_threshold` was sent; `null` otherwise |
| `inline_text` | Raw document text; populated only when `token_count <= inline_threshold` |

**Error codes:**
- `401` — missing or invalid `X-API-Key`
- `403` — path outside `INGEST_ROOT` or collection not in `ALLOWED_COLLECTIONS`
- `404` — file not found
- `413` — file exceeds `MAX_FILE_SIZE_MB`
- `502` — upstream document-chunker failure
- `500` — embedding or Qdrant upsert failure

---

### `POST /sync`

Reconciles the filesystem under `INGEST_ROOT` with what's stored in Qdrant:
- Detects files present on disk but missing in Qdrant (returns them as `new_files` for the caller to ingest).
- Deletes orphaned chunks for files that no longer exist on disk.

The scan runs in a background thread, is TTL-cached (30 s) and timeout-protected. Returns `503` under heavy load.

**Request:**
```json
{"collection": "documents"}
```

**Response:**
```json
{
  "collection": "documents",
  "new_files": ["/data/new_doc.pdf"],
  "deleted_chunks": 12
}
```

---

### `GET /health`

Returns `{"status": "ok"}`.

---

## Environment

**Required:**

| Variable | Description |
|---|---|
| `DOCUMENT_CHUNKER_URL` | document-chunker `/chunk` endpoint |
| `QDRANT_HOST` | Qdrant hostname |
| `API_KEY` | Shared secret for `X-API-Key` header |
| `QDRANT_API_KEY` | Qdrant service API key (required in production) |

**Optional (defaults shown):**

| Variable | Default | Description |
|---|---|---|
| `QDRANT_PORT` | `6333` | Qdrant port |
| `INGEST_ROOT` | `/data` | Filesystem root for allowed file paths |
| `ALLOWED_COLLECTIONS` | `documents` | Comma-separated list of valid collection names |
| `CHUNK_SIZE` | `512` | Default chunk size forwarded to document-chunker |
| `OVERLAP` | `1` | Default overlap forwarded to document-chunker |
| `DENSE_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | fastembed dense model |
| `SPARSE_MODEL_NAME` | `Qdrant/bm25` | fastembed sparse model |
| `BATCH_SIZE` | `16` | Embedding batch size |
| `UPSERT_BATCH_SIZE` | `16` | Qdrant upsert batch size |
| `SCROLL_LIMIT` | `1000` | Qdrant scroll page size (sync + orphan cleanup) |
| `MAX_FILE_SIZE_MB` | `50` | Maximum file size accepted for ingest |
| `APP_PORT` | `8002` | Host port |
| `DEBUG_ERRORS` | `false` | Include exception details in HTTP error bodies (non-prod only) |
| `DEBUG_LOG_FILE` | — | Write DEBUG-level logs to this file; console stays at INFO |

---

## Security

- **API key** — validated on every request with constant-time comparison. Service refuses to start if `API_KEY` is not configured.
- **Path traversal** — all file paths are resolved and validated to be inside `INGEST_ROOT`.
- **Collection safety** — only collections listed in `ALLOWED_COLLECTIONS` are accepted.
- **Error leakage** — internal exceptions are logged server-side only; HTTP responses use generic messages. Enable `DEBUG_ERRORS=true` only in non-production environments.

---

## Implementation notes

- **Idempotency** — point IDs are derived deterministically from `file_path` + chunk index, so retrying an ingest will upsert (not duplicate) existing points.
- **Upsert resilience** — batches are retried with backoff; per-batch failures are reported in `failed_batches` without aborting the whole request.
- **Sync safety** — filesystem scans are bounded (200k file limit), run in a thread pool, TTL-cached for 30 s, and timeout-protected at 60 s.
- **Token counting** — inline threshold uses a fast whitespace-split approximation (`len(text.split())`), not a tokenizer. Suitable for rough gating; do not rely on it for exact context-window management.

---

## Troubleshooting

- **Auth errors** — call protected endpoints with `X-API-Key: <API_KEY>` header.
- **Docker Compose interpolation errors** — ensure all required vars are in `.env`.
- **Inline text always returned** — check that `inline_threshold` is not set too high relative to your document sizes.
- **503 on sync** — filesystem scan is overloaded or timed out; retry after a few seconds.
