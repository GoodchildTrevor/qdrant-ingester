# qdrant-ingester

FastAPI microservice that receives document file paths, sends them to [document-chunker](https://github.com/GoodchildTrevor/document-chunker) for parsing and chunking, embeds the resulting chunks, and upserts them into Qdrant.

## Architecture

```
  caller
    │
    ├── POST /ingest  ──►  document-chunker /chunk  ──►  embed  ──►  Qdrant
    └── POST /sync   ──►  diff filesystem vs Qdrant  ──►  delete orphans
```

## API

### `POST /ingest`

Full pipeline for a single file: chunk → embed → upsert.

**Request body (JSON):**
```json
{
  "collection": "my_collection",
  "file_path": "/data/documents/report.pdf",
  "chunk_size": 512,
  "overlap": 1
}
```
`chunk_size` and `overlap` are optional — defaults come from env.

**Response:**
```json
{
  "collection": "my_collection",
  "file_name": "report.pdf",
  "chunks_upserted": 42
}
```

### `POST /sync`

Compare a list of current file paths against what is stored in Qdrant. Returns new paths (not yet ingested) and deletes orphaned chunks for files that no longer exist.

**Request body (JSON):**
```json
{
  "collection": "my_collection",
  "current_file_paths": ["/data/documents/a.pdf", "/data/documents/b.docx"]
}
```

**Response:**
```json
{
  "collection": "my_collection",
  "new_files": ["/data/documents/b.docx"],
  "deleted_chunks": 17
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Run

```bash
cp .env.example .env
# Set DOCUMENT_CHUNKER_URL, QDRANT_HOST
docker compose up --build
```

## Environment variables

`DOCUMENT_CHUNKER_URL` and `QDRANT_HOST` are **required**.

| Variable | Default | Description |
|---|---|---|
| `DOCUMENT_CHUNKER_URL` | *(required)* | URL of the document-chunker `/chunk` endpoint |
| `QDRANT_HOST` | *(required)* | Qdrant hostname |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `DENSE_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | fastembed dense model |
| `SPARSE_MODEL_NAME` | `Qdrant/bm25` | fastembed sparse (BM25) model |
| `BATCH_SIZE` | `16` | Embedding batch size |
| `UPSERT_BATCH_SIZE` | `16` | Qdrant upsert batch size |
| `SCROLL_LIMIT` | `1000` | Qdrant scroll page size |
| `CHUNK_SIZE` | `512` | Default chunk_size forwarded to document-chunker |
| `OVERLAP` | `1` | Default overlap forwarded to document-chunker |
| `APP_PORT` | `8002` | Host port (docker-compose only) |
