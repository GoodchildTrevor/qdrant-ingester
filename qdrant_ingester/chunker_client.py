import logging
from pathlib import Path

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
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    with open(file_path, "rb") as fh:
        file_bytes = fh.read()

    filename = file_path.name
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            chunker_url,
            data={"chunk_size": chunk_size, "overlap": overlap},
            files={"file": (filename, file_bytes)},
        )
    response.raise_for_status()
    return ChunkResponse.model_validate(response.json())
