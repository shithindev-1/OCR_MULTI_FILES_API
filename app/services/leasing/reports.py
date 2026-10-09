"""The leasing response and its per-document report entries, and tracing pipeline results
back to the uploads they came from."""

import logging
import time
from typing import Any

from app.schemas.ocr import (
    fields_for_tenant_type,
)

logger = logging.getLogger("uae_ocr.leasing")


def new_report(item: dict[str, Any]) -> dict[str, Any]:
    """One `documents` entry per source, before anything has been read. `document_type` stays
    "unknown" until the source passes validation; a merged front/back pair is folded into a single
    entry with both sources only once the whole request has succeeded."""
    return {
        "document_type": "unknown",
        "document_type_is_correct": False,
        "merged": False,
        "sources": [{
            "source": item["source"],
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "document_type": item["document_type"],
            "pages": [],
            "ocr_file": None,
        }],
        "error": None,
    }


def failed_response(
    *, tenant_type: str, document_refnumber: str, documents: list[dict[str, Any]], message: str,
    started_at: float,
) -> dict[str, Any]:
    """A "failed" response in the normal shape, with `message` on every document - for a request
    that could not be processed at all, so a caller always gets the usual structure back."""
    reports = []
    for document in documents:
        report = new_report({
            "source": document.get("source") or "",
            "document_id": document.get("document_id") or "",
            "document_name": document.get("document_name") or "",
            "document_type": document.get("document_type") or "",
        })
        report["error"] = message
        reports.append(report)
    return build_response(
        status="failed", tenant_type=tenant_type, document_refnumber=document_refnumber,
        started_at=started_at, documents=reports,
    )


def first_error_message(response: dict[str, Any]) -> str:
    errors = response.get("errorInfo") or []
    if errors:
        return errors[0].get("DocumentErrorToShow") or errors[0].get("DocumentError") or "Validation failed"
    return "Validation failed"


def build_response(
    *, status: str, tenant_type: str, document_refnumber: str, started_at: float,
    documents: list[dict[str, Any]], form_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one response shape both endpoints return, success or failure. `form_data` keys are the
    lowercase names (tenant_name_en, ... for an individual; company_name_en, ... for a company)
    rather than the pipeline's own Tenant_Name_En spelling."""
    fields, keys = fields_for_tenant_type(tenant_type)
    form_data = {
        key: (form_data or {}).get(field)
        for field, key in zip(fields, keys)
    }
    elapsed = round(time.perf_counter() - started_at, 2)
    return {
        "status": status,
        "tenant_type": tenant_type,
        "document_refnumber": document_refnumber,
        "form_data": form_data,
        "missing": [key for key, value in form_data.items() if value is None],
        "processing_time": f"{elapsed:g} sec",
        "documents": documents,
    }


def fold_merged_pairs(
    documents_report: list[dict[str, Any]], merged_pairs: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    """Replaces each merged front/back pair's two entries with one `merged: true` entry carrying
    both sources, placed where the earlier of the two was uploaded."""
    folded: dict[int, dict[str, Any]] = {}
    absorbed: set[int] = set()
    for front_index, back_index, document_type in merged_pairs:
        first, second = sorted((front_index, back_index))
        folded[first] = {
            "document_type": document_type,
            "document_type_is_correct": True,
            "merged": True,
            "sources": documents_report[front_index]["sources"] + documents_report[back_index]["sources"],
            "error": None,
        }
        absorbed.add(second)
    return [
        folded.get(index, report)
        for index, report in enumerate(documents_report)
        if index not in absorbed
    ]


def indices_named(names: str, index_by_name: dict[str, int], fallback: list[int]) -> list[int]:
    """Maps an errorInfo DocumentName back to upload positions. The pipeline names a merged pair
    "<front> + <back>" and an ambiguous group "<a>, <b>, ..."; anything unrecognised falls back to
    `fallback` so an error is never silently dropped."""
    if names in index_by_name:
        return [index_by_name[names]]
    parts = [part.strip() for chunk in names.split(" + ") for part in chunk.split(", ")]
    found = [index_by_name[part] for part in parts if part in index_by_name]
    return found or fallback


def merged_indices(document: dict[str, Any], index_by_name: dict[str, int]) -> list[int]:
    if document.get("is_merged_pair"):
        front_name, _, back_name = document["document_name"].partition(" + ")
        return [index_by_name[front_name], index_by_name[back_name]]
    return [index_by_name[document["document_name"]]]


def unconfirmed_type_error(document_name: str, document_type: str, types_present: list[str]) -> str:
    """For a type-detected document whose second reading did not agree with the first. Kept short
    and plain: the two readings only matter to the logs, which already carry them."""
    logger.warning(
        "Detected type not confirmed document=%s detected=%s extraction_read=%s",
        document_name, document_type, types_present,
    )
    return f"We couldn't clearly identify '{document_name}'. Please upload a clearer copy."


def add_error(report: dict[str, Any], message: str) -> None:
    """Adds a short message to a document's error, once."""
    existing = report["error"]
    if existing and message in existing:
        return
    report["error"] = f"{existing} {message}" if existing else message
