# qdrant-ingester

FastAPI service that ingests documents (by path), sends them to a document-chunker for chunking, embeds chunk lemmas (dense + sparse), and upserts points into Qdrant.

## Quick start

1. Copy example env:
```bash
cp .env.example .env
```

2. Fill required variables in .env (see "Environment" below).

3. Run with Docker Compose (recommended):
```bash
docker compose up --build
```

Or run locally:
```bash
python -m uvicorn qdrant_ingester.main:app --host 0.0.0.0 --port 8002
```

## API

- POST /ingest
  - Full pipeline: chunk → embed → upsert.
  - Request (JSON):
```json
{
  "collection": "my_collection",
  "file_path": "/data/documents/report.pdf",
  "chunk_size": 512,
  "overlap": 1
}
```
  - Behavior:
    - Returns 413 if file exceeds configured MAX_FILE_SIZE_MB.
    - Returns 502 for upstream document-chunker failures.
    - Returns 500 for pipeline/internal failures (embedding, upsert errors).
    - Authentication: X-API-Key header required (see ENV).
    - Collection is validated against configured allowed_collections.

- POST /sync
  - Reconciles filesystem under INGEST_ROOT with what’s stored in Qdrant:
    - Detects new files for ingest.
    - Deletes orphaned chunks.
  - The endpoint is protected and bounded:
    - Filesystem scan runs in a background thread, is timeout-protected and cached to avoid I/O storms.
    - On overload/timeout the endpoint returns 503 (retry later).

- GET /health
  - Returns {"status": "ok"}.

## Responses

- /ingest returns structured status with partial/failure details:
  - status: "success" | "partial" | "failed"
  - chunks_total, chunks_upserted, chunks_failed, failed_batches (batch-level errors)

## Security & hardening notes

- API key:
  - Service expects a configured API_KEY and validates requests with a constant-time comparison.
  - If not set the service refuses to start (configuration validation).
- Qdrant:
  - Qdrant client is created with QDRANT_API_KEY when configured.
  - docker-compose enforces providing QDRANT_API_KEY for the qdrant container.
- Errors:
  - Internal exception details are logged server-side only; API responses use generic error messages to avoid information leakage.
  - Set `DEBUG_ERRORS=true` (non-prod only) to include the real exception message in the HTTP response body — useful when you can't easily tail container logs.
  - Set `DEBUG_LOG_FILE=/path/to/debug.log` to write full DEBUG-level logs to a separate file. The console handler stays at INFO, so main output remains clean without the extra noise.
- Collection safety:
  - Allowed collection names are validated and only collections in allowed_collections are accepted for ingest/sync to prevent accidental data deletion.

## Environment (important / required)

Minimum required:
- DOCUMENT_CHUNKER_URL (required) — document-chunker /chunk endpoint
- QDRANT_HOST (required) — Qdrant hostname
- API_KEY (required) — shared API key in X-API-Key header
- QDRANT_API_KEY (required for docker-compose + recommended for production)

Common optional (defaults shown):
- QDRANT_PORT=6333
- DENSE_MODEL_NAME=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- SPARSE_MODEL_NAME=Qdrant/bm25
- BATCH_SIZE=16
- UPSERT_BATCH_SIZE=16
- SCROLL_LIMIT=1000
- CHUNK_SIZE=512
- OVERLAP=1
- INGEST_ROOT=/data
- MAX_FILE_SIZE_MB=50
- APP_PORT=8002
- ALLOWED_COLLECTIONS=("documents",) — validated names only
- DEBUG_ERRORS=false — set to `true` to include exception details in HTTP error bodies (non-prod only)
- DEBUG_LOG_FILE= — if set, full DEBUG-level logs go to this file; console stays at INFO

Note: docker-compose has been updated to require API_KEY, DOCUMENT_CHUNKER_URL, QDRANT_HOST and QDRANT_API_KEY; starting compose without these will error and prevents insecure defaults.

## Implementation notes (operators)

- Idempotency: point IDs deterministic (derived from file_path and chunk index) to avoid duplicates on retries.
- Upserts: done in batches with per-batch retries and backoff; final failed batches are reported to the client.
- Filesystem sync:
  - Scans are bounded, run in a thread, cached for a short TTL and timeboxed to avoid blocking the event loop or causing excessive I/O.
  - On heavy load the sync endpoint degrades gracefully with 503 responses.
- Logging: full exception traces are logged server-side for diagnostics; API responses are intentionally generic.

## Troubleshooting

- If docker compose interpolation errors occur, ensure required env vars are present in your environment or .env file.
- To test authentication, call a protected endpoint with header:
  X-API-Key: <API_KEY>

## Contributing / Development

- Run tests and linters in your environment before opening PRs.
- Pay attention to configuration validation; the app validates required secrets on startup.