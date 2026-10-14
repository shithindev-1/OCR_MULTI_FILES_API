"""The type-detection response: form_data and missing per tenant, and one entry per document."""

import posixpath
import time
from typing import Any
from urllib.parse import urlparse

from app.schemas.ocr import detect_fields_for_tenant_type


def build_detect_response(
    *, status: str, tenant_type: str, document_refnumber: str, started_at: float,
    documents: list[dict[str, Any]], tenants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The type-detection response. `tenants` is [{tenant_id, status, form_data}] with form_data in
    the pipeline's field names; each becomes one entry of the `form_data` and `missing` arrays."""
    fields, keys = detect_fields_for_tenant_type(tenant_type)
    form_data, missing = [], []
    for tenant in tenants or []:
        values = {key: tenant["form_data"].get(field) for field, key in zip(fields, keys)}
        form_data.append({"tenant_id": tenant["tenant_id"], "status": tenant["status"], **values})
        missing.append({"tenant_id": tenant["tenant_id"], "fields": [key for key, value in values.items() if value is None]})
    return {
        "status": status,
        "tenant_type": tenant_type,
        "document_refnumber": document_refnumber,
        "form_data": form_data,
        "missing": missing,
        "processing_time": f"{round(time.perf_counter() - started_at, 2):g} sec",
        "documents": documents,
    }


def document_entry(
    *, document_type: str, status: str, sources: list[dict[str, Any]], error: str | None,
    merged: bool = False, tenant_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tenant_ids": tenant_ids or [], "document_type": document_type, "status": status,
        "merged": merged, "sources": sources, "error": error,
    }


def failed_detect_response(
    message: str, *, tenant_type: str, document_refnumber: str, sources: list[str], started_at: float,
) -> dict[str, Any]:
    """A "failed" response in the normal shape with `message` on every file - for a job that ended
    without a result of its own (a missing mandatory document, or a problem on our side)."""
    documents = []
    for source in sources:
        name = posixpath.basename(urlparse(source).path) or source
        row = {"source": source, "file_name": name, "pages": [], "ocr_file": None}
        documents.append(document_entry(document_type="unknown", status="invalid", sources=[row], error=message))
    return build_detect_response(
        status="failed", tenant_type=tenant_type, document_refnumber=document_refnumber,
        started_at=started_at, documents=documents,
    )
