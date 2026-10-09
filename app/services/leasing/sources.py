"""Fetching a request's documents: from an allowed URL, or from the configured blob container."""

import logging
import posixpath
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.schemas.leasing import SourceDocument
from app.services.blob_storage import BlobStorage, BlobStorageError
from app.services.validation import ALLOWED_FILES

logger = logging.getLogger("uae_ocr.leasing")


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}


def safe_name(value: str) -> str:
    return _UNSAFE_NAME.sub("_", value.strip())[:80] or "document"


def source_name(source: SourceDocument) -> str:
    """The file name a person recognises - the link's last part - for use in messages."""
    return posixpath.basename(urlparse(source.source).path) or source.document_name


async def download_source(client: httpx.AsyncClient, source: SourceDocument) -> tuple[bytes, str, str]:
    """Downloads one `source` URL. Returns (content, filename, content_type).

    filename/content_type are derived the same way a browser upload's would be, so the downloaded
    bytes can be handed straight to `validate_document_request` unchanged: the URL's own last path
    segment is the filename when it has one of the four supported extensions, falling back to the
    response's Content-Type header, then to the document's own declared name as a last resort.
    """
    host = (urlparse(source.source).hostname or "").lower()
    if host not in settings.allowed_source_host_list:
        raise HTTPException(
            status_code=400,
            detail=f"'{source_name(source)}' cannot be downloaded from this location. Please upload the file "
                   "directly, or use a link to the company's file storage.",
        )

    limit = settings.max_file_size_mb * 1024 * 1024
    too_large = HTTPException(
        status_code=413,
        detail=f"'{source_name(source)}' is too large. The largest file allowed is {settings.max_file_size_mb} MB.",
    )
    try:
        # No redirects: a redirect could lead anywhere, outside the allowed hosts.
        async with client.stream("GET", source.source, follow_redirects=False, timeout=60.0) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            received = 0
            # Stop reading the moment the limit is passed, instead of buffering an unbounded body.
            async for chunk in response.aiter_bytes():
                received += len(chunk)
                if received > limit:
                    raise too_large
                chunks.append(chunk)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        logger.warning("Source download failed source=%s error=%s", host, exc)
        raise HTTPException(
            status_code=502,
            detail=f"We could not download '{source_name(source)}'. Please check the link and try again.",
        ) from exc

    content = b"".join(chunks)

    url_name = Path(urlparse(source.source).path).name or source.document_name
    extension = Path(url_name).suffix.lower()
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()

    if extension in ALLOWED_FILES:
        filename = url_name
        if content_type not in ALLOWED_FILES.values():
            content_type = ALLOWED_FILES[extension]
    elif content_type in _CONTENT_TYPE_EXTENSIONS:
        filename = f"{Path(url_name).stem or safe_name(source.document_name)}{_CONTENT_TYPE_EXTENSIONS[content_type]}"
    else:
        # Nothing usable was reported at all - let validate_document_request reject it with its own
        # clear "only JPG/JPEG/PNG/WEBP/PDF" message rather than guessing.
        filename = url_name or f"{source.document_name}.bin"

    return content, filename, content_type


async def download_blob(
    storage: BlobStorage, blob_path: str, source: SourceDocument,
) -> tuple[bytes, str, str]:
    """Fetches one source from blob storage. Returns (content, filename, content_type) exactly
    like `download_source`, so the rest of the batch cannot tell the two apart."""
    try:
        content, stored_type = await storage.download(
            blob_path, max_bytes=settings.max_file_size_mb * 1024 * 1024,
        )
    except BlobStorageError as exc:
        logger.warning("Blob source download failed path=%s error=%s", blob_path, exc)
        message = (
            f"We could not find '{source_name(source)}'. Please check the file path and try again."
            if exc.not_found else
            f"We could not download '{source_name(source)}' right now. Please try again in a few minutes."
        )
        raise HTTPException(status_code=502, detail=message) from exc
    filename = posixpath.basename(blob_path) or f"{safe_name(source.document_name)}.bin"
    extension = posixpath.splitext(filename)[1].lower()
    # The extension wins over whatever Content-Type the blob was stored with (often a generic
    # application/octet-stream); validate_document_request still checks the real bytes.
    content_type = ALLOWED_FILES.get(extension) or (stored_type or "").split(";")[0].strip().lower()
    return content, filename, content_type
