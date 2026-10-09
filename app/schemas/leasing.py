"""Request and response models for the /ocr/leasing endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    source: str = Field(..., description="A direct, downloadable URL to the document file.")
    document_id: str
    document_type: str
    document_name: str


class LeasingOcrRequest(BaseModel):
    tenant_type: str
    document_refnumber: str
    sources: list[SourceDocument]


class SourceReport(BaseModel):
    """One original upload or URL behind a reported document."""

    source: str = Field(..., description="The source URL, or the uploaded file's name.")
    document_id: str
    document_name: str | None
    document_type: str = Field(..., description="The document type exactly as it was sent.")
    pages: list[int] = Field(..., description="Page numbers read from this source; [] if it was never read.")
    ocr_file: str | None = Field(
        None,
        description="Full blob URL of the uploaded cropped/rotated result (<name>_ocr.<ext>, beside the "
                    "original); null when the source is not in blob storage or nothing was uploaded.",
    )


class DocumentReport(BaseModel):
    """What happened to one processed document - a merged front/back pair is a single entry."""

    document_type: str = Field(..., description='Normalised type, e.g. "national_id"; "unknown" before validation.')
    document_type_is_correct: bool
    merged: bool
    sources: list[SourceReport]
    error: str | None = Field(..., description="This document's own problem, or null if it passed.")


class LeasingOcrResponse(BaseModel):
    status: Literal["success", "failed"]
    tenant_type: str
    document_refnumber: str
    form_data: dict[str, Any] = Field(..., description="Every extracted field; all null on failure.")
    missing: list[str] = Field(..., description="The form_data keys that are null.")
    processing_time: str = Field(..., examples=["14.72 sec"])
    documents: list[DocumentReport]
