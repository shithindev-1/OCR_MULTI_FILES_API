"""Request and response models for the type-detection endpoints, /ocr/leasing/detect."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectSource(BaseModel):
    """One file link. Nothing else is accepted: the document's type is found, never declared."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="A blob path in the configured container, or an allowed URL.")


class DetectOcrRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_type: str
    document_refnumber: str
    sources: list[DetectSource]


class DetectedSource(BaseModel):
    source: str = Field(..., description="The source URL or path, or the uploaded file's name.")
    file_name: str
    pages: list[int] = Field(..., description="Pages of that file this document was found on.")
    ocr_file: str | None = Field(None, description="Blob URL of the uploaded cropped/rotated result, if any.")


class DetectedDocument(BaseModel):
    """One document found in the uploads - a front/back pair from two files is a single entry."""

    tenant_ids: list[str] = Field(
        ...,
        description="The tenant(s) this document belongs to, matching form_data[].tenant_id. A company's "
                    "trade licence belongs to every tenant (signatory) of that company; [] when it fits none.",
    )
    document_type: str = Field(..., description='The type found, e.g. "national_id"; "other" when unsupported.')
    status: Literal["valid", "invalid", "ignored", "not_checked"] = Field(
        ...,
        description="valid: passed its own checks. invalid: see error. ignored: not a supported type, "
                    "never checked. not_checked: another file could not be read, so nothing was checked.",
    )
    merged: bool
    sources: list[DetectedSource]
    error: str | None


class TenantMissing(BaseModel):
    tenant_id: str
    fields: list[str] = Field(..., description="The form_data keys that are null for this tenant.")


class DetectOcrResponse(BaseModel):
    status: Literal["success", "failed"] = Field(..., description="success only when every tenant succeeded.")
    tenant_type: str
    document_refnumber: str
    form_data: list[dict[str, Any]] = Field(
        ...,
        description="One entry per tenant (a person; for a company, one per company and signatory): "
                    "tenant_id, status (success/failed), then every field - all null when that tenant failed.",
    )
    missing: list[TenantMissing]
    processing_time: str
    documents: list[DetectedDocument]
