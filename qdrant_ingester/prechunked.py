r"""Support for pre-chunked JSON/JSONL input.

Sometimes the caller already has ready-made chunks (e.g. an ERP export of
PL/SQL objects where each function/procedure is its own chunk). In that
case the document-chunker is skipped: the file itself is the chunk list.

Supported formats:

* ``.jsonl`` / ``.ndjson`` — one JSON object per line (blank lines ignored);
* ``.json`` — a JSON array of chunk objects, a ``{"chunks": [...]}`` wrapper
  or a single chunk object.

Each chunk object must contain a text field: ``content`` (preferred),
``raw`` or ``text``. An optional ``lemmas`` field can carry pre-lemmatized
text used for embedding (falls back to the raw text). All other fields are
kept as chunk metadata and end up in the Qdrant point payload, e.g.::

    {"chunk_id": "standalone_function.AAA_CALC_SUM_FACT_PAY.acf2a0959bb9",
     "package_name": null,
     "procedure_name": "AAA_CALC_SUM_FACT_PAY",
     "kind": "standalone_function",
     "content": "CREATE OR REPLACE FUNCTION ...",
     "signature": null,
     "start_line": 1,
     "end_line": 16,
     "source_file": "erp_export/functions/AAA_CALC_SUM_FACT_PAY.sql",
     "file_size": 425}

A nested format is also supported, where the text and the metadata live
inside a ``payload`` dict (e.g. a Hyperion Plan export)::

    {"id": 6656971675031269153,
     "payload": {"type": "dimension_definition",
                 "app": "ECG_C0st",
                 "db": "Plan1",
                 "dimension": "ECG_CostAccount",
                 "source_path": "./HP_export\\ECG_CostAccount",
                 "fields": {...},
                 "text": "Измерение: ECG_CostAccount (приложение ...)"}}

The ``payload`` is only unwrapped when the object itself has no text field.
Then the text is taken from ``payload`` (``content``/``raw``/``text``,
first match wins), an optional ``payload["lemmas"]`` is used for embedding
when the top level has none, and the remaining fields are merged into the
chunk metadata: top-level keys first (so ``id`` is kept), then payload keys
under names not already taken. An object with neither top-level text nor
text inside ``payload`` is rejected.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from qdrant_ingester.schemas import ChunkResponse, ChunkSchema

logger = logging.getLogger(__name__)

# File extensions treated as pre-chunked input when the request does not
# force the mode explicitly.
PRECHUNKED_EXTENSIONS = frozenset({".jsonl", ".ndjson", ".json"})

# Keys that may carry the chunk text (first match wins).
_CONTENT_KEYS = ("content", "raw", "text")


class PrechunkedParseError(ValueError):
    """Raised when a pre-chunked file cannot be parsed into chunks."""


def is_prechunked(path: Path, override: bool | None = None) -> bool:
    """Decide whether *path* should be treated as pre-chunked input.

    An explicit *override* from the request wins; otherwise the decision is
    made from the file extension.
    """
    if override is not None:
        return override
    return path.suffix.lower() in PRECHUNKED_EXTENSIONS


def _first_text(obj: dict) -> str | None:
    """Return the value of the first *obj* key that holds a text."""
    for key in _CONTENT_KEYS:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return None


def _as_chunk(index: int, obj: object) -> ChunkSchema:
    if not isinstance(obj, dict):
        raise PrechunkedParseError(f"chunk #{index} is not a JSON object")

    raw = _first_text(obj)
    if raw is not None:
        # Flat format: everything but the text/lemmas keys is metadata.
        # An existing "payload" stays a plain metadata field — no unwrap.
        lemmas = obj.get("lemmas")
        if not isinstance(lemmas, str) or not lemmas:
            lemmas = raw  # no pre-lemmatized text: embed the raw content

        meta = {k: v for k, v in obj.items() if k not in _CONTENT_KEYS and k != "lemmas"}
        return ChunkSchema(raw=raw, lemmas=lemmas, meta=meta)

    # No top-level text: maybe a nested {"id": ..., "payload": {...}} object.
    payload = obj.get("payload")
    nested_raw = _first_text(payload) if isinstance(payload, dict) else None
    if nested_raw is None:
        raise PrechunkedParseError(
            f"chunk #{index} has no text field (expected one of "
            f"{list(_CONTENT_KEYS)} on the top level or inside 'payload')"
        )

    # Lemmas: the top level wins over the nested payload.
    lemmas = obj.get("lemmas")
    if not isinstance(lemmas, str) or not lemmas:
        lemmas = payload.get("lemmas")
    if not isinstance(lemmas, str) or not lemmas:
        lemmas = nested_raw  # no pre-lemmatized text: embed the raw content

    # Metadata: top-level keys first (so e.g. "id" is kept), then the
    # payload fields under names not already taken by the top level.
    meta = {
        k: v
        for k, v in obj.items()
        if k not in _CONTENT_KEYS and k not in ("lemmas", "payload")
    }
    for k, v in payload.items():
        if k not in _CONTENT_KEYS and k != "lemmas" and k not in meta:
            meta[k] = v
    return ChunkSchema(raw=nested_raw, lemmas=lemmas, meta=meta)


def _objects_from_json(text: str) -> list[object]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PrechunkedParseError(f"invalid JSON: {exc}") from exc

    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        items: object = data["chunks"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        raise PrechunkedParseError(
            "expected a JSON array of chunks, a {'chunks': [...]} object "
            "or a single chunk object"
        )
    return items


def _objects_from_jsonl(text: str) -> list[object]:
    items: list[object] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise PrechunkedParseError(f"invalid JSON on line {lineno}: {exc}") from exc
    return items


def parse_prechunked_file(path: Path) -> ChunkResponse:
    """Parse a pre-chunked JSON/JSONL file into a ChunkResponse.

    Behaves like a local replacement for the document-chunker call, so the
    rest of the ingest pipeline (embedding, payload, upsert) stays unchanged.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PrechunkedParseError(f"file is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise PrechunkedParseError(f"cannot read file: {exc}") from exc

    # A leading '[' means the whole file is one JSON document (array);
    # otherwise treat it as JSON Lines.
    if path.suffix.lower() == ".json" or text.lstrip().startswith("["):
        items = _objects_from_json(text)
    else:
        items = _objects_from_jsonl(text)

    if not items:
        raise PrechunkedParseError("no chunks found in file")

    chunks = [_as_chunk(i, obj) for i, obj in enumerate(items)]

    stat = path.stat()
    logger.info("Parsed %d pre-chunked entries from %s", len(chunks), path)
    return ChunkResponse(
        file_name=path.name,
        file_format=f"prechunked/{path.suffix.lower().lstrip('.')}",
        creation_date=datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
        modification_date=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        chunks=chunks,
    )
