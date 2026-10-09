"""Per-request token accounting across every Gemini call, written to usage.log."""

import json
import logging
from contextvars import ContextVar
from typing import Any

from app.core.logging import log_timestamps

# Per-request usage accumulator. Every model call this service makes adds a row, so one request
# can be totalled across all of its calls and pipelines - which no single log file could do before,
# since hits.log only ever recorded the extraction call. A ContextVar rather than an attribute
# because the service is a process-wide singleton while these totals belong to one request; each
# request runs in its own task, so each gets its own copy.
_usage_records: ContextVar[list[dict[str, Any]] | None] = ContextVar("ocr_usage", default=None)


# The calls a request can make, named for what they are for rather than for the method that makes
# them, because these strings are the keys in usage.log:
#   extraction -> GeminiService.extract        (the KYC read; the extraction pipeline)
#   crop       -> GeminiService.detect_box_corners  (finds the document to crop)
#   rotation   -> GeminiService.detect_orientations (decides which way is up)
#   validation -> GeminiService.verify_document / verify_documents_cross / verify_document_pair
#                 (self-checks, cross-checks, and front/back pairing confirmation)
CALL_EXTRACTION = "extraction"


CALL_CROP = "crop"


CALL_ROTATION = "rotation"


CALL_VALIDATION = "validation"


#   segmentation -> GeminiService.segment_documents (type-detection endpoint only)
CALL_SEGMENTATION = "segmentation"


# crop and rotation are the two halves of the crop/rotate pipeline, and are subtotalled together.
CROP_ROTATE_CALLS = (CALL_CROP, CALL_ROTATION)


def start_usage_tracking() -> None:
    """Begin a fresh per-request tally. Call once, at the start of a request."""
    _usage_records.set([])


def record_usage(call: str, input_tokens: Any, output_tokens: Any) -> None:
    records = _usage_records.get()
    if records is None:
        return  # not inside a tracked request (a script, a test) - nothing to tally
    records.append({
        "call": call,
        "input_tokens": int(input_tokens) if isinstance(input_tokens, int) else 0,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, int) else 0,
    })


def _totals(records: list[dict[str, Any]]) -> dict[str, int]:
    input_tokens = sum(r["input_tokens"] for r in records)
    output_tokens = sum(r["output_tokens"] for r in records)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "calls": len(records),
    }


def usage_summary() -> dict[str, Any]:
    """Per-call and per-pipeline token totals for the current request.

    Each call gets its own key, so a reader never has to know which pipeline made which call or
    correlate separate log lines by timestamp. A call that did not run - crop and rotation when
    the pipeline is switched off, or when the document was rejected at validation - reports zeros
    with calls=0 rather than being omitted, so the shape of the record is the same every time and
    can be parsed without conditionals.
    """
    records = _usage_records.get() or []
    by_call = {name: [r for r in records if r["call"] == name]
               for name in (CALL_EXTRACTION, CALL_CROP, CALL_ROTATION, CALL_VALIDATION, CALL_SEGMENTATION)}
    summary: dict[str, Any] = {name: _totals(rows) for name, rows in by_call.items()}
    summary["crop_rotate_pipeline"] = _totals(
        [r for r in records if r["call"] in CROP_ROTATE_CALLS]
    )
    summary["request_total"] = _totals(records)
    return summary


def log_usage_summary(request_id: str, status: str, detection_enabled: bool) -> None:
    """Write this request's usage to usage.log as one JSON object."""
    summary = usage_summary()
    usage_logger.info(json.dumps({
        **log_timestamps(),
        "request_id": request_id,
        "status": status,
        "detection_enabled": detection_enabled,
        **summary,
    }))
    return summary


usage_logger = logging.getLogger("uae_ocr.usage")
