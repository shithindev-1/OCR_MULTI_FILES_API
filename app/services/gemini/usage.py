"""Per-request token accounting across every Gemini call, written to usage.log."""

import functools
import json
import logging
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from app.core.logging import log_timestamps

# Per-request usage accumulator. Every model call this service makes adds a row, so one request
# can be totalled across all of its calls and pipelines - which no single log file could do before,
# since hits.log only ever recorded the extraction call. A ContextVar rather than an attribute
# because the service is a process-wide singleton while these totals belong to one request; each
# request runs in its own task, so each gets its own copy.
_usage_records: ContextVar[list[dict[str, Any]] | None] = ContextVar("ocr_usage", default=None)

# Which document(s) - (name, document type) - the model call being made is for, so document-usage.log
# can say what each document type costs. Set around the work done for a document; a call made for
# several (a file's split, a cross-check) is shared among them equally.
_usage_for: ContextVar[tuple[tuple[str, str], ...]] = ContextVar("ocr_usage_for", default=())
# Names that stand for several documents - a file whose split call ran before its documents were
# known - and the documents they turned out to hold. One dict per request.
_usage_groups: ContextVar[dict[str, list[tuple[str, str]]] | None] = ContextVar("ocr_usage_groups", default=None)


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
    _usage_groups.set({})


def documents_named(name: Any, document_type: Any) -> list[tuple[str, str]]:
    """The documents a pipeline document stands for: a merged front/back pair is named
    "<front> + <back>" and counts for both sides."""
    return [(part.strip(), str(document_type or "")) for part in str(name or "document").split(" + ")]


@contextmanager
def usage_for(*documents: tuple[str, str]) -> Iterator[None]:
    """Charges the model calls made inside this block to `documents`."""
    token = _usage_for.set(tuple(documents))
    try:
        yield
    finally:
        _usage_for.reset(token)


def charged_to(documents_of: Callable[..., Iterable[tuple[str, str]]]) -> Callable[..., Any]:
    """Decorator for an async function: its model calls are charged to `documents_of(*args, **kwargs)`."""
    def wrap(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        async def run(*args: Any, **kwargs: Any) -> Any:
            with usage_for(*documents_of(*args, **kwargs)):
                return await function(*args, **kwargs)
        return run
    return wrap


def share_usage(name: str, documents: list[tuple[str, str]]) -> None:
    """Calls charged to `name` (a file's split, made before its documents were known) are shared
    among `documents`."""
    groups = _usage_groups.get()
    if groups is not None:
        groups[name] = list(documents)


def record_usage(call: str, input_tokens: Any, output_tokens: Any) -> None:
    records = _usage_records.get()
    if records is None:
        return  # not inside a tracked request (a script, a test) - nothing to tally
    records.append({
        "call": call,
        "input_tokens": int(input_tokens) if isinstance(input_tokens, int) else 0,
        "output_tokens": int(output_tokens) if isinstance(output_tokens, int) else 0,
        "for": _usage_for.get(),
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
    _log_document_usage(request_id, status)
    return summary


_CALLS = (CALL_SEGMENTATION, CALL_EXTRACTION, CALL_VALIDATION, CALL_CROP, CALL_ROTATION)


def document_usage() -> list[dict[str, Any]]:
    """This request's tokens per document. A call made for one document is charged to it whole; a
    call made for several (a file's split, a cross-check, a front/back check) is shared among them
    equally, so every document's tokens add up to the request's total. `calls` counts every call a
    document took part in, shared or not. A call made for no document is charged to "unassigned"."""
    groups = _usage_groups.get() or {}
    usage: dict[tuple[str, str], dict[str, list[float]]] = {}
    for record in _usage_records.get() or []:
        owners: list[tuple[str, str]] = []
        for name, document_type in record.get("for") or ():
            for owner in groups.get(name) or [(name, document_type)]:
                if owner not in owners:
                    owners.append(owner)
        owners = owners or [("unassigned", "unassigned")]
        for owner in owners:
            row = usage.setdefault(owner, {call: [0, 0.0, 0.0] for call in _CALLS})
            totals = row.setdefault(record["call"], [0, 0.0, 0.0])
            totals[0] += 1
            totals[1] += record["input_tokens"] / len(owners)
            totals[2] += record["output_tokens"] / len(owners)
    documents = []
    for (name, document_type), row in usage.items():
        calls = {
            call: {"calls": totals[0], "input_tokens": round(totals[1]), "output_tokens": round(totals[2])}
            for call, totals in row.items()
        }
        input_tokens = sum(entry["input_tokens"] for entry in calls.values())
        output_tokens = sum(entry["output_tokens"] for entry in calls.values())
        documents.append({
            "document": name,
            "document_type": document_type or "unknown",
            **calls,
            "document_total": {
                "calls": sum(entry["calls"] for entry in calls.values()),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        })
    return documents


def _log_document_usage(request_id: str, status: str) -> None:
    """Writes this request's tokens per document to document-usage.log, one JSON object per document."""
    stamps = log_timestamps()
    for document in document_usage():
        document_usage_logger.info(json.dumps({**stamps, "request_id": request_id, "status": status, **document}))


usage_logger = logging.getLogger("uae_ocr.usage")
document_usage_logger = logging.getLogger("uae_ocr.document_usage")
