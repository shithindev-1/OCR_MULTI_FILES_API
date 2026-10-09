"""X-API-Key authentication for every /api/v1 route."""

import logging
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.core.config import settings

logger = logging.getLogger("uae_ocr")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject the request unless X-API-Key matches one of `API_KEYS`.

    With `API_KEYS` empty, authentication is off - for local testing only.
    """
    keys = settings.api_key_list
    if not keys:
        return
    # compare_digest against every key, so response timing reveals nothing about which one is close.
    if api_key and any(secrets.compare_digest(api_key, key) for key in keys):
        return
    logger.warning("Request rejected reason=invalid_api_key")
    raise HTTPException(status_code=401, detail="Your API key is missing or not valid. Please check the X-API-Key header.", headers={"WWW-Authenticate": "APIKey"})
