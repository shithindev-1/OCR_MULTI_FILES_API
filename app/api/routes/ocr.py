"""POST /ocr - the original multipart endpoint, one merged OcrResponse per request."""

import logging
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.core.config import settings
from app.services.gemini import start_usage_tracking
from app.services.ocr_pipeline import run_ocr_batch

logger = logging.getLogger("uae_ocr")

router = APIRouter(tags=["ocr"])


@router.post("/ocr")
async def extract_ocr(
    request: Request,
    file: Annotated[list[UploadFile], File(...)],
    documentName: Annotated[list[str], Form(...)],
    documentType: Annotated[list[str], Form(...)],
) -> Any:
    """Extract one or more documents from a single multipart request into one merged response.

    `file`, `documentName`, and `documentType` are repeated fields paired by position, so every
    document carries the type its own dropdown selected. Every document is extracted and
    type-checked first; an incomplete National ID/passport/visa is not rejected immediately but held
    to see whether another uploaded file of the same type supplies its missing side - if so, and
    Gemini confirms the two genuinely are the same document, they are merged into one before
    validation continues (see `resolve_incomplete_pairs` in app.services.ocr_pipeline). Every resulting document then runs
    self-verification, and any problem anywhere - a type mismatch, a document nothing else can
    complete, an ambiguous pairing, a rejected pairing, or a self-verification finding - stops the
    whole request immediately: the response is exactly the same shape a single document has always
    returned, `data` empty and the reason in `errorInfo`, never a partial result for the documents
    that were fine. Only once every uploaded document has individually passed does cross-verification
    run across all of them (skipped entirely for a single document, since there is nothing to compare
    it against) - another failure there stops the request the same way. Only once the whole batch has
    cleared every one of those gates, including cross-verification, does the crop/rotate pipeline
    run, for every document at once: an error anywhere in the batch means none of them is ever
    cropped or rotated. Only then are every document's own fields folded into that one response: a
    National ID's National_Id, a passport's Passport_Number, a visa's Visa_Number, all landing in the
    same `data` object beside each other, exactly the schema a single-document upload has always
    used - never a per-document breakdown, never a new key.
    """
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    if not all(char.isalnum() or char in "-_." for char in request_id) or len(request_id) > 128:
        request_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    start_usage_tracking()
    # Per-request override of the crop/rotate and verification pipelines, so either can be switched
    # off for testing without restarting the service. Absent header means the configured default.
    def _header_enabled(name: str, default: bool) -> bool:
        value = (request.headers.get(name) or "").strip().lower()
        return value not in {"off", "false", "0", "no"} if value else default

    detection_enabled = _header_enabled("x-detection", settings.detection_enabled)
    validation_enabled = _header_enabled("x-validation", settings.validation_enabled)
    logger.info(
        "OCR request started request_id=%s model=%s detection=%s validation=%s documents=%d",
        request_id, settings.gemini_model, "on" if detection_enabled else "off",
        "on" if validation_enabled else "off", len(file),
    )
    # A best-effort identity for the error handlers until the first document is in flight.
    request.state.document_name = documentName[0] if documentName else "unknown"
    request.state.document_filename = request.state.document_name

    if not file:
        logger.warning("OCR file rejection request_id=%s reason=file_count", request_id)
        raise HTTPException(status_code=400, detail="Please upload at least one document.")
    if not len(documentName) == len(documentType) == len(file):
        logger.warning(
            "OCR file rejection request_id=%s reason=field_count files=%d names=%d types=%d",
            request_id, len(file), len(documentName), len(documentType),
        )
        raise HTTPException(
            status_code=400,
            detail="Every file needs its own documentName and documentType. Please check that none are missing.",
        )


    return await run_ocr_batch(
        request=request,
        request_id=request_id,
        file=file,
        documentName=documentName,
        documentType=documentType,
        detection_enabled=detection_enabled,
        validation_enabled=validation_enabled,
        started_at=started_at,
    )
