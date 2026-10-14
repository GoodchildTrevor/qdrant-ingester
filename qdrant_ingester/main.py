import json
import logging
import secrets
import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from qdrant_ingester.config import (
    get_settings,
    get_qdrant_client,
    get_dense_model,
    get_sparse_model,
    Settings,
)
from qdrant_ingester.schemas import (
    IngestRequest, IngestResponse,
    SyncRequest, SyncResponse,
    PreviewRequest, PreviewResponse, ChunkPreview,
)
from qdrant_ingester.chunker_client import fetch_chunks
from qdrant_ingester.prechunked import (
    PrechunkedParseError,
    is_prechunked,
    parse_prechunked_file,
)
from qdrant_ingester.embedder import embed_texts
from qdrant_ingester.loader import upsert_data, sync_file_paths, delete_orphaned_chunks

logger = logging.getLogger(__name__)

# Keys that callers must not override via extra_payload
_RESERVED_PAYLOAD_KEYS = frozenset({
    "name", "file_path", "file_format", "creation_date", "modification_date"
})

# Dense vector dimensions by known model names
_DENSE_DIMS: dict[str, int] = {
    "intfloat/multilingual-e5-large": 1024,
    "intfloat/multilingual-e5-base": 768,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "bge-m3": 1024,
}
_DENSE_DIMS_DEFAULT = 768

_STATIC_DIR = Path(__file__).parent / "static"


def _merge_payload(base: dict, extra: dict) -> dict:
    """Merge extra_payload into base, raising 422 if any reserved key is present."""
    overlap = _RESERVED_PAYLOAD_KEYS & extra.keys()
    if overlap:
        raise HTTPException(
            status_code=422,
            detail=f"extra_payload cannot override reserved keys: {sorted(overlap)}",
        )
    return {**base, **extra}


def setup_logging(settings: Settings) -> None:
    fmt = "[%(asctime)s] %(levelname)s %(name)s - %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    if settings.debug_log_file:
        try:
            fh = logging.FileHandler(settings.debug_log_file)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not open debug log file %r: %s", settings.debug_log_file, exc
            )
        else:
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(fmt))
            root.addHandler(fh)
            logger.debug("Debug log file: %s", settings.debug_log_file)


async def _create_collection(
    client: AsyncQdrantClient, settings: Settings, collection: str
) -> None:
    dense_dim = _DENSE_DIMS.get(settings.dense_model_name, _DENSE_DIMS_DEFAULT)
    await client.create_collection(
        collection_name=collection,
        vectors_config={settings.dense_vector_config: VectorParams(
            size=dense_dim,
            distance=Distance.COSINE,
        )},
        sparse_vectors_config={settings.sparse_vector_config: SparseVectorParams(
            index=SparseIndexParams(on_disk=False),
        )},
    )
    logger.info(
        "Created collection '%s' (dense_dim=%d, dense='%s', sparse='%s').",
        collection, dense_dim,
        settings.dense_vector_config,
        settings.sparse_vector_config,
    )


async def _ensure_collection(
    client: AsyncQdrantClient, settings: Settings, collection: str
) -> None:
    """Create the collection in Qdrant if it does not exist yet."""
    existing = {c.name for c in (await client.get_collections()).collections}
    if collection in existing:
        return
    await _create_collection(client, settings, collection)


async def ensure_collections(settings: Settings) -> None:
    """Create all allowed_collections in Qdrant if they do not already exist."""
    client = get_qdrant_client()

    existing = {c.name for c in (await client.get_collections()).collections}

    for collection in settings.allowed_collections:
        if collection in existing:
            logger.info("Collection '%s' already exists, skipping creation.", collection)
            continue
        await _create_collection(client, settings, collection)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    try:
        await ensure_collections(settings)
    except Exception:
        logger.exception("Failed to ensure Qdrant collections on startup")
    yield


app = FastAPI(title="qdrant-ingester", version="0.2.0", lifespan=lifespan)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _error_detail(generic: str, exc: Exception, settings: Settings) -> str:
    if settings.debug_errors:
        return f"{generic}: {exc}"
    return generic


@app.get("/health")
def health():
    return {"status": "ok"}


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    settings = get_settings()
    secret = settings.api_key.get_secret_value() if settings.api_key else ""
    if not secret:
        raise HTTPException(status_code=500, detail="API key not configured")
    if not api_key or not secrets.compare_digest(api_key, secret):
        raise HTTPException(status_code=401, detail="Invalid API key")


def resolve_path(settings: Settings, path_str: str) -> Path:
    root = Path(settings.ingest_root).resolve()
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root):
        logger.warning(
            "Rejected path %r: not relative to ingest_root=%r", str(candidate), str(root)
        )
        raise HTTPException(status_code=403, detail="File path not allowed")
    return candidate


def require_allowed_collection(collection: str, settings: Settings) -> None:
    if collection not in settings.allowed_collections:
        logger.warning(
            "Rejected collection %r: not in allowed_collections=%s",
            collection,
            list(settings.allowed_collections),
        )
        raise HTTPException(status_code=403, detail="Collection is not allowed")


async def _ingest_file(settings: Settings, request: IngestRequest, file_path: Path) -> IngestResponse:
    """Shared ingest pipeline used by /ingest and /ingest_upload.

    The file must already exist at *file_path* (under INGEST_ROOT).
    """
    # Validate extra_payload keys early, before any external calls (chunker, embedder, Qdrant),
    # so a rejected request never depends on how many chunks the document happens to produce.
    _merge_payload({}, request.extra_payload)

    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    # Path guard: ensure resolved path is under ingest_root
    allowed_root = Path(settings.ingest_root).resolve()
    if not file_path.is_relative_to(allowed_root):
        logger.warning(
            "Rejected path %r: not under ingest_root=%r (collection=%r, file_name=%r)",
            str(file_path),
            str(allowed_root),
            request.collection,
            request.file_name,
        )
        raise HTTPException(status_code=403, detail="path not allowed")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        size_mb = file_path.stat().st_size / (1024 * 1024)
    except Exception:
        size_mb = None
    if not getattr(settings, "disable_file_size_limit", False) and size_mb is not None and size_mb > settings.max_file_size_mb:
        raise HTTPException(status_code=413, detail=f"File too large ({size_mb:.1f} MB)")

    # --- obtain chunks ----------------------------------------------------
    # Pre-chunked JSON/JSONL files are parsed locally, document-chunker is skipped.
    prechunked_flag = is_prechunked(file_path, request.prechunked)
    if prechunked_flag:
        logger.info("Ingesting pre-chunked file %s (document-chunker skipped)", file_path)
        try:
            chunk_resp = parse_prechunked_file(file_path)
        except PrechunkedParseError as e:
            logger.warning("Invalid pre-chunked file %s: %s", file_path, e)
            raise HTTPException(status_code=422, detail=f"invalid pre-chunked file: {e}")
    else:
        chunk_size = request.chunk_size or settings.chunk_size
        overlap = request.overlap or settings.overlap
        try:
            chunk_resp = await fetch_chunks(
                chunker_url=settings.document_chunker_url,
                file_path=file_path,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        except Exception as e:
            logger.exception("Chunker call failed for %s", file_path)
            raise HTTPException(status_code=502, detail=_error_detail("document-chunker error", e, settings))

    if not chunk_resp.chunks:
        logger.warning("No chunks returned for %s, skipping upsert.", file_path)
        return IngestResponse(
            collection=request.collection,
            file_name=chunk_resp.file_name,
            status="success",
            partial=False,
            message=None,
            chunks_total=0,
            chunks_upserted=0,
            chunks_failed=0,
            failed_batches=[],
            prechunked=prechunked_flag,
        )

    # --- inline_threshold check: return raw text if document is small enough ---
    if request.inline_threshold is not None:
        full_text = "\n\n".join(ch.raw for ch in chunk_resp.chunks)
        token_count = len(full_text.split())
        if token_count <= request.inline_threshold:
            logger.info(
                "Document %s is below inline_threshold (%d <= %d), returning inline text.",
                file_path, token_count, request.inline_threshold,
            )
            return IngestResponse(
                collection=request.collection,
                file_name=chunk_resp.file_name,
                status="success",
                partial=False,
                message=None,
                chunks_total=len(chunk_resp.chunks),
                chunks_upserted=0,
                chunks_failed=0,
                failed_batches=[],
                token_count=token_count,
                inline_text=full_text,
                prechunked=prechunked_flag,
            )
    else:
        token_count = None

    lemmas = [ch.lemmas for ch in chunk_resp.chunks]
    try:
        dense_embs, sparse_embs = await embed_texts(
            texts=lemmas,
            dense_model=dense_model,
            sparse_model=sparse_model,
            batch_size=settings.batch_size,
        )
    except Exception as e:
        logger.exception("Embedding failed for %s", file_path)
        raise HTTPException(status_code=500, detail=_error_detail("Embedding error", e, settings))

    chunks_as_dicts = [
        {"raw": ch.raw, "lemmas": ch.lemmas, "meta": ch.meta}
        for ch in chunk_resp.chunks
    ]
    base_payload = {
        "name": chunk_resp.file_name,
        "file_path": str(file_path),
        "file_format": chunk_resp.file_format,
        "creation_date": chunk_resp.creation_date,
        "modification_date": chunk_resp.modification_date,
    }
    base_payload = _merge_payload(base_payload, request.extra_payload)

    # Make sure the target collection exists (it may be missing, e.g. when it
    # was added to ALLOWED_COLLECTIONS or removed from Qdrant manually).
    try:
        await _ensure_collection(client, settings, request.collection)
    except Exception as e:
        logger.exception("Failed to ensure collection '%s'", request.collection)
        raise HTTPException(status_code=500, detail=_error_detail("Qdrant collection error", e, settings))

    try:
        metrics = await upsert_data(
            client=client,
            collection_name=request.collection,
            dense_vector_config=settings.dense_vector_config,
            sparse_vector_config=settings.sparse_vector_config,
            dense_embeddings=dense_embs,
            sparse_embeddings=sparse_embs,
            base_payload=base_payload,
            chunks=chunks_as_dicts,
            batch_size=settings.upsert_batch_size,
        )
    except Exception as e:
        logger.exception("Upsert failed for %s", file_path)
        raise HTTPException(status_code=500, detail=_error_detail("Qdrant upsert error", e, settings))

    status_str = "success"
    if metrics.get("chunks_failed", 0) > 0:
        status_str = "partial"
    if metrics.get("chunks_upserted", 0) == 0 and metrics.get("chunks_failed", 0) > 0:
        status_str = "failed"

    failed = metrics.get("chunks_failed", 0)
    failed_batches = metrics.get("failed_batches", [])
    partial_flag = failed > 0 and metrics.get("chunks_upserted", 0) > 0
    message = None
    if failed:
        message = f"{failed} chunks failed in {len(failed_batches)} failed batch(es)"
    return IngestResponse(
        collection=request.collection,
        file_name=chunk_resp.file_name,
        status=status_str,
        partial=partial_flag,
        message=message,
        chunks_total=metrics.get("chunks_total", 0),
        chunks_upserted=metrics.get("chunks_upserted", 0),
        chunks_failed=failed,
        failed_batches=failed_batches,
        token_count=token_count,
        inline_text=None,
        prechunked=prechunked_flag,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    _: None = Depends(require_api_key),
):
    settings = get_settings()
    require_allowed_collection(request.collection, settings)

    # Build file_path from collection + file_name
    allowed_root = Path(settings.ingest_root).resolve()
    file_path = (allowed_root / request.collection / request.file_name).resolve()

    return await _ingest_file(settings, request, file_path)


@app.post("/ingest_upload", response_model=IngestResponse)
async def ingest_upload(
    file: UploadFile = File(..., description="Document or pre-chunked JSON/JSONL file"),
    collection: str = Form(...),
    prechunked: bool | None = Form(None),
    extra_payload: str | None = Form(None),
    inline_threshold: int | None = Form(None),
    _: None = Depends(require_api_key),
):
    """Upload a file (multipart) and ingest it in one call.

    The file is stored under INGEST_ROOT/<collection>/<file name> and the
    standard pipeline runs afterwards (document-chunker, or local parsing
    for pre-chunked JSON/JSONL, then embedding and upsert).
    """
    settings = get_settings()
    require_allowed_collection(collection, settings)

    if extra_payload is not None:
        try:
            parsed_extra = json.loads(extra_payload)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=422, detail=f"extra_payload is not valid JSON: {e}")
        if not isinstance(parsed_extra, dict):
            raise HTTPException(status_code=422, detail="extra_payload must be a JSON object")
    else:
        parsed_extra = {}

    # Strip any directory components from the uploaded file name
    safe_name = Path((file.filename or "").replace("\\", "/")).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=422, detail="Invalid file name")

    root = Path(settings.ingest_root).resolve()
    dest_dir = root / collection
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / safe_name).resolve()
    if not dest.is_relative_to(root):
        raise HTTPException(status_code=403, detail="path not allowed")

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    disable_limit = getattr(settings, "disable_file_size_limit", False)
    try:
        with open(dest, "wb") as out:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                if not disable_limit and dest.stat().st_size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (> {settings.max_file_size_mb} MB)",
                    )
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as e:
        dest.unlink(missing_ok=True)
        logger.exception("Failed to store uploaded file %s", dest)
        raise HTTPException(status_code=500, detail=_error_detail("Failed to store uploaded file", e, settings))
    finally:
        await file.close()

    logger.info("Stored uploaded file %r (%d bytes)", str(dest), dest.stat().st_size)

    request = IngestRequest(
        collection=collection,
        file_name=safe_name,
        prechunked=prechunked,
        extra_payload=parsed_extra,
        inline_threshold=inline_threshold,
    )
    return await _ingest_file(settings, request, dest)


def _scan_files_bounded(root: Path, limit: int = 200_000) -> set[Path]:
    out = set()
    for i, p in enumerate(root.rglob("*"), start=1):
        if p.is_file():
            out.add(p.resolve())
        if i >= limit:
            raise RuntimeError(f"sync scan limit exceeded: {limit}")
    return out

_sync_lock = asyncio.Lock()
_sync_cache: dict[str, object] = {"ts": 0.0, "paths": set()}
SYNC_CACHE_TTL_SECONDS = 30
SYNC_SCAN_TIMEOUT_SECONDS = 60

async def get_current_paths_cached(root: Path) -> set[Path]:
    now = time.time()
    if now - float(_sync_cache["ts"]) <= SYNC_CACHE_TTL_SECONDS:
        return set(_sync_cache["paths"])

    async with _sync_lock:
        now = time.time()
        if now - float(_sync_cache["ts"]) <= SYNC_CACHE_TTL_SECONDS:
            return set(_sync_cache["paths"])

        paths = await asyncio.wait_for(
            asyncio.to_thread(_scan_files_bounded, root, 200_000),
            timeout=SYNC_SCAN_TIMEOUT_SECONDS,
        )
        _sync_cache["ts"] = time.time()
        _sync_cache["paths"] = set(paths)
        return set(paths)


@app.get("/files/{collection}")
async def list_files(
    collection: str,
    _: None = Depends(require_api_key),
):
    settings = get_settings()
    require_allowed_collection(collection, settings)

    root = Path(settings.ingest_root) / collection
    if not root.exists():
        return {"collection": collection, "files": []}

    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            stat = p.stat()
            files.append({
                "name": p.name,
                "path": str(p.relative_to(root)),  # relative to collection dir
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": stat.st_mtime,
            })
    return {"collection": collection, "files": files}


@app.post("/sync", response_model=SyncResponse)
async def sync(
    request: SyncRequest,
    _ : None = Depends(require_api_key),
):
    settings = get_settings()
    require_allowed_collection(request.collection, settings)
    client = get_qdrant_client()

    if not getattr(settings, "ingest_root", None):
        raise HTTPException(status_code=500, detail="INGEST_ROOT must be configured for sync")
    allowed_root = settings.ingest_root
    try:
        current_paths = await get_current_paths_cached(allowed_root)
    except (TimeoutError, asyncio.TimeoutError, RuntimeError):
        raise HTTPException(
            status_code=503,
            detail="Filesystem scan is overloaded; retry later",
        )

    try:
        new_paths, deleted_paths = await sync_file_paths(
            client=client,
            collection=request.collection,
            current_file_paths=current_paths,
            scroll_limit=settings.scroll_limit,
        )
    except Exception as e:
        logger.exception("Sync failed")
        raise HTTPException(status_code=500, detail=_error_detail("Sync error", e, settings))

    deleted_count = 0
    if deleted_paths:
        try:
            deleted_count = await delete_orphaned_chunks(
                client=client,
                collection_name=request.collection,
                deleted_file_paths=deleted_paths,
                scroll_limit=settings.scroll_limit,
            )
        except Exception as e:
            logger.exception("Orphan cleanup failed")
            raise HTTPException(status_code=500, detail=_error_detail("Cleanup error", e, settings))

    return SyncResponse(
        collection=request.collection,
        new_files=[str(p) for p in new_paths],
        deleted_chunks=deleted_count,
    )


@app.get("/", include_in_schema=False)
def root_ui():
    return FileResponse(_STATIC_DIR / "index.html")


if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


@app.post("/preview", response_model=PreviewResponse)
async def preview(
    request: PreviewRequest,
    _: None = Depends(require_api_key),
):
    settings = get_settings()
    require_allowed_collection(request.collection, settings)

    file_path = Path(settings.ingest_root) / request.collection / request.file_name
    file_path = file_path.resolve()

    # Reuse existing path guard
    allowed_root = Path(settings.ingest_root).resolve()
    if not file_path.is_relative_to(allowed_root):
        raise HTTPException(status_code=403, detail="path not allowed")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    if is_prechunked(file_path, request.prechunked):
        logger.info("Previewing pre-chunked file %s (document-chunker skipped)", file_path)
        try:
            chunk_resp = parse_prechunked_file(file_path)
        except PrechunkedParseError as e:
            logger.warning("Invalid pre-chunked file %s: %s", file_path, e)
            raise HTTPException(status_code=422, detail=f"invalid pre-chunked file: {e}")
    else:
        chunk_resp = await fetch_chunks(
            chunker_url=settings.document_chunker_url,
            file_path=file_path,
            chunk_size=request.chunk_size or settings.chunk_size,
            overlap=request.overlap or settings.overlap,
        )

    return PreviewResponse(
        collection=request.collection,
        file_name=chunk_resp.file_name,
        file_format=chunk_resp.file_format,
        chunks_total=len(chunk_resp.chunks),
        chunks=[
            ChunkPreview(
                index=i,
                raw=ch.raw,
                lemmas=ch.lemmas,
                char_count=len(ch.raw),
                word_count=len(ch.raw.split()),
            )
            for i, ch in enumerate(chunk_resp.chunks)
        ],
    )
