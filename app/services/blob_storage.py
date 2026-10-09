"""Azure Blob Storage access for the /ocr/leasing endpoints: fetch source files, upload results.

Configured by AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER. The SDK client is
synchronous, so every call runs in a worker thread rather than blocking the event loop. Neither the
connection string nor any signed URL is ever logged or returned to a caller.
"""

import asyncio
import logging
from functools import lru_cache
from urllib.parse import quote, unquote, urlparse

from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from app.core.config import settings

logger = logging.getLogger("uae_ocr.blob")


class BlobStorageError(Exception):
    """A blob could not be read or written. The message is safe to show to the caller."""

    def __init__(self, message: str, *, not_found: bool = False) -> None:
        super().__init__(message)
        self.not_found = not_found


def _reason(exc: AzureError) -> str:
    # The SDK's message carries RequestId/Time/ErrorCode lines after the human-readable first one.
    return (str(exc).strip().splitlines() or [type(exc).__name__])[0]


class BlobStorage:
    def __init__(self, connection_string: str, container: str) -> None:
        service = BlobServiceClient.from_connection_string(connection_string)
        self._container = service.get_container_client(container)
        # e.g. https://<account>.blob.core.windows.net/<container> - no credentials in it.
        self.container_url = self._container.url.rstrip("/")

    def blob_path(self, source: str) -> str | None:
        """The blob path `source` names inside this container, or None if it points elsewhere.

        Accepts either a path relative to the container ("ApplicationFiles/.../Passport.jpg") or a
        full URL into this same container, with or without a query string.
        """
        if source.lower().startswith(("http://", "https://")):
            parsed = urlparse(source)
            base = urlparse(self.container_url)
            prefix = base.path.rstrip("/") + "/"
            if parsed.netloc.lower() != base.netloc.lower() or not parsed.path.startswith(prefix):
                return None
            path = unquote(parsed.path[len(prefix):])
        else:
            path = source.strip().lstrip("/")
        return path or None

    def url(self, path: str) -> str:
        """The blob's full URL, e.g. https://<account>.blob.core.windows.net/<container>/<path>.
        Carries no credentials, so it is safe to return to a caller."""
        return f"{self.container_url}/{quote(path, safe='/')}"

    async def download(self, path: str, *, max_bytes: int) -> tuple[bytes, str | None]:
        """Returns (content, content_type as stored on the blob)."""

        def _run() -> tuple[bytes, str | None]:
            downloader = self._container.download_blob(path)
            if downloader.size > max_bytes:
                raise BlobStorageError("exceeds the configured size limit")
            return downloader.readall(), downloader.properties.content_settings.content_type

        try:
            return await asyncio.to_thread(_run)
        except AzureError as exc:
            logger.warning("Blob download failed path=%s error=%s", path, type(exc).__name__)
            raise BlobStorageError(_reason(exc), not_found=isinstance(exc, ResourceNotFoundError)) from exc

    async def warm_up(self) -> None:
        """Opens the pooled connection to the storage account before the first request needs it.

        Reads the container's properties only - no blob is listed or read.
        """
        await asyncio.to_thread(self._container.get_container_properties)

    async def upload(self, path: str, content: bytes, content_type: str) -> None:
        """Writes `content` to `path`, replacing any blob already there (a re-run of the request)."""

        def _run() -> None:
            self._container.upload_blob(
                path, content, overwrite=True, content_settings=ContentSettings(content_type=content_type),
            )

        try:
            await asyncio.to_thread(_run)
        except AzureError as exc:
            logger.warning("Blob upload failed path=%s error=%s", path, type(exc).__name__)
            raise BlobStorageError(_reason(exc)) from exc
        logger.info("Blob uploaded path=%s bytes=%d content_type=%s", path, len(content), content_type)


@lru_cache
def get_blob_storage() -> BlobStorage | None:
    """The shared client, or None when blob storage is not configured."""
    if not settings.azure_storage_connection_string or not settings.azure_storage_container:
        return None
    return BlobStorage(settings.azure_storage_connection_string, settings.azure_storage_container)
