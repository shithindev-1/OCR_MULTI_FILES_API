"""Pipeline stop signal and the app-wide exception handlers.

/ocr answers every error in its own errorInfo response shape; every other route keeps FastAPI's
standard `{"detail": ...}` body.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.post_processing import error_response

logger = logging.getLogger("uae_ocr")

API_V1_PREFIX = "/api/v1"
OCR_PATH = f"{API_V1_PREFIX}/ocr"
LEASING_PATH = f"{API_V1_PREFIX}/ocr/leasing"
# Paths served before every route moved under /api/v1 - a 404 on one names its new location.
_LEGACY_PREFIXES = ("/ocr",)


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _moved_to(path: str) -> str | None:
    if any(path == prefix or path.startswith(prefix + "/") for prefix in _LEGACY_PREFIXES):
        return API_V1_PREFIX + path
    return None


class ValidationStopped(Exception):
    """Raised the moment any document fails type-matching, completeness, or self-verification.

    Carries the exact response the request should answer with - always blank `data` plus the
    reason in `errorInfo` - so the whole request can stop in one place (`run_ocr_batch`'s catch)
    instead of every caller of `validate_and_extract_one` needing its own copy of that logic.
    One document's problem is the whole request's problem: there is only ever one response object,
    so there is nowhere a failure could be reported that would let the rest keep going.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response


async def ocr_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    path = request.url.path
    if path != OCR_PATH:
        detail = exc.detail
        moved_to = _moved_to(path) if exc.status_code == 404 else None
        if moved_to:
            detail = f"This address has changed. Please use {moved_to} instead."
        elif exc.status_code == 404 and exc.detail == "Not Found":
            detail = "This address does not exist. Please check the URL and try again."
        elif exc.status_code == 405:
            detail = (
                f"This address does not accept {request.method} requests. "
                "Please check whether the request should be GET or POST."
            )
        log = logger.error if exc.status_code >= 500 else logger.warning
        log(
            "Request rejected method=%s path=%s status_code=%d client=%s detail=%s%s",
            request.method, path, exc.status_code, _client_ip(request), exc.detail,
            f" moved_to={moved_to}" if moved_to else "",
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": detail}, headers=exc.headers)
    logger.warning("OCR request rejected status_code=%d detail=%s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            document_name=getattr(request.state, "document_name", "unknown"),
            document_filename=getattr(request.state, "document_filename", "unknown"),
            message=exc.detail,
        ),
        headers=exc.headers,
    )


_UNEXPECTED_ERROR = "Something went wrong on our side. Please try again in a few minutes."


def _missing_fields_message(exc: RequestValidationError) -> str:
    """One readable sentence naming the request fields that are missing or not valid.

    Built from each error's field name only - never its submitted value - so it can never echo a
    file's bytes or anything else the caller sent, and it is always plain, serialisable text.
    """
    names: list[str] = []
    for error in exc.errors():
        loc = [part for part in error.get("loc", ()) if isinstance(part, str) and part != "body"]
        name = ".".join(loc) or "request body"
        if name not in names:
            names.append(name)
    listed = ", ".join(names) if names else "the request"
    return f"Some required information is missing or not valid: {listed}. Please check the request and try again."


async def ocr_request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path != OCR_PATH:
        fields = [{"loc": err.get("loc"), "type": err.get("type")} for err in exc.errors()]
        logger.warning(
            "Request rejected method=%s path=%s status_code=422 client=%s reason=malformed_request fields=%s",
            request.method, request.url.path, _client_ip(request), fields,
        )
        return JSONResponse(status_code=422, content={"detail": _missing_fields_message(exc)})
    # loc/type only, never "input" — a field's raw submitted value could be arbitrarily large
    # or, for the file field, binary content, and has no place in a text log.
    fields = [{"loc": err.get("loc"), "type": err.get("type")} for err in exc.errors()]
    logger.warning("OCR request rejected status_code=422 reason=malformed_request fields=%s", fields)
    return JSONResponse(
        status_code=422,
        content=error_response(
            document_name=getattr(request.state, "document_name", "unknown"),
            document_filename=getattr(request.state, "document_filename", "unknown"),
            message=_missing_fields_message(exc),
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request.headers.get("x-request-id", "unknown")
    logger.exception(
        "Unhandled error method=%s path=%s request_id=%s error_type=%s",
        request.method, request.url.path, request_id, type(exc).__name__,
    )
    if request.url.path != OCR_PATH:
        return JSONResponse(status_code=500, content={"detail": _UNEXPECTED_ERROR})
    return JSONResponse(
        status_code=500,
        content=error_response(
            document_name=getattr(request.state, "document_name", "unknown"),
            document_filename=getattr(request.state, "document_filename", "unknown"),
            message=_UNEXPECTED_ERROR,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    # Starlette's class, so routing 404/405s are caught too - FastAPI's HTTPException subclasses it.
    app.add_exception_handler(StarletteHTTPException, ocr_http_exception_handler)
    app.add_exception_handler(RequestValidationError, ocr_request_validation_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
