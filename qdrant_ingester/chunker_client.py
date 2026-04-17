import logging
from pathlib import Path
import json

import httpx

from qdrant_ingester.schemas import ChunkResponse

logger = logging.getLogger(__name__)


async def fetch_chunks(
    chunker_url: str,
    file_path: Path,
    chunk_size: int,
    overlap: int,
    timeout: float = 300.0,
) -> ChunkResponse:
    """
    Send a file to document-chunker and return the parsed ChunkResponse.

    Stream the file by using httpx.AsyncClient.stream and passing an open
    file object to files= so the request body is not built entirely in memory.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    filename = file_path.name
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(file_path, "rb") as fh:
            # Use client.stream to avoid buffering the entire response or request
            async with client.stream(
                "POST",
                chunker_url,
                data={"chunk_size": chunk_size, "overlap": overlap},
                files={"file": (filename, fh, "application/octet-stream")},
            ) as response:
                response.raise_for_status()
                raw = await response.aread()
    return ChunkResponse.model_validate(json.loads(raw.decode("utf-8")))