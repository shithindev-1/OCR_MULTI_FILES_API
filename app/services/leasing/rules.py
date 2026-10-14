"""Request rules checked before anything is downloaded or read: complete rows, unique names,
and the mandatory document for the tenant type."""

from typing import Any

from fastapi import HTTPException

from app.schemas.ocr import (
    COMPANY_ALTERNATIVE_DOCUMENT_TYPE,
    COMPANY_REQUIRED_DOCUMENT_TYPE,
    INDIVIDUAL_REQUIRED_DOCUMENT_TYPE,
    canonical_document_type,
    is_company,
)


def ensure_documents_complete(documents: list[dict[str, Any]], *, check_source: bool) -> None:
    """Rejects a request where any document has an empty field, before anything is downloaded, read
    or started as a job - so the caller learns straight away, in plain words, what to fill in.

    `documents` are dicts with document_id, document_type, document_name and - when `check_source`
    - source. A value of only spaces counts as empty.
    """
    for position, document in enumerate(documents, start=1):
        name = (document.get("document_name") or "").strip()
        label = f"'{name}'" if name else f"document {position}"
        if not name:
            message = (
                f"Document {position} has no document_name. Please give each document a name and try again."
            )
        elif check_source and not (document.get("source") or "").strip():
            message = f"The file link for {label} is empty. Please add the document's link or file path and try again."
        elif not (document.get("document_type") or "").strip():
            message = (
                f"The document type for {label} is empty. Please add it (for example passport or national_id) "
                "and try again."
            )
        elif not (document.get("document_id") or "").strip():
            message = f"The document_id for {label} is empty. Please add it and try again."
        else:
            continue
        raise HTTPException(status_code=422, detail=message)


def ensure_unique_document_names(names: list[str]) -> None:
    """Rejects a request that reuses a document_name - names are how a merged front/back result and
    every per-document error are traced back to the right upload, so a repeat would be ambiguous."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"Each document needs a different document_name. This name is used more than once: "
                   f"{', '.join(duplicates)}.",
        )


def require_mandatory_document(
    tenant_type: str, document_types: list[str], *, accept_initial_approval: bool = False,
) -> None:
    """Rejects a submission missing the one document its tenant type cannot go without: a trade
    licence for a company, a National ID for an individual.

    Judged on the types the caller declared, before anything is downloaded or read, so a missing
    document costs nothing. A document declared as the required type that turns out to be something
    else is still caught later, by the normal type check on that document.
    """
    declared = {
        canonical_document_type(value.strip().lower().replace(" ", "_").replace("-", "_"))
        for value in document_types
    }
    if is_company(tenant_type) and accept_initial_approval:
        # Type-detection endpoints: a trade licence or an initial approval.
        if not declared & {COMPANY_REQUIRED_DOCUMENT_TYPE, COMPANY_ALTERNATIVE_DOCUMENT_TYPE}:
            raise HTTPException(
                status_code=400,
                detail="A trade licence or initial approval is required for a company. Please upload the "
                       "company's trade licence or initial approval and try again.",
            )
    elif is_company(tenant_type):
        if COMPANY_REQUIRED_DOCUMENT_TYPE not in declared:
            raise HTTPException(
                status_code=400,
                detail="A trade licence is required for a company. Please upload the company's trade licence "
                       "and try again.",
            )
    elif INDIVIDUAL_REQUIRED_DOCUMENT_TYPE not in declared:
        raise HTTPException(
            status_code=400,
            detail="A National ID is required for an individual. Please upload the tenant's National ID "
                   "and try again.",
        )
