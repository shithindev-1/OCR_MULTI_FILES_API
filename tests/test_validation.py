from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image
from pypdf import PdfWriter

from app.core.config import Settings
from app.services.validation import enforce_rate_limit, validate_document_request


def _settings(**overrides: object) -> Settings:
    values = {"gemini_api_key": "test", "gemini_model": "test-model", **overrides}
    return Settings(**values)


def _upload(filename: str, content_type: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(content), headers={"content-type": content_type})


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), "white").save(output, format="PNG")
    return output.getvalue()


def _pdf(page_count: int) -> bytes:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=100, height=100)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_rejects_mismatched_mime_and_extension() -> None:
    upload = _upload("document.png", "image/png", b"not-a-png")
    with pytest.raises(HTTPException, match="doesn't look like a genuine"):
        validate_document_request(
            file=upload,
            content=b"not-a-png",
            document_name="National ID",
            document_type="national_id",
            settings=_settings(),
        )


@pytest.mark.parametrize(
    "document_type",
    ["passport", "national_id", "visa", "ejari_certificate", "trade_license"],
)
def test_single_page_pdf_is_never_rejected_on_page_count(document_type: str) -> None:
    # Regression: a passport whose front and back are both on one scanned page was rejected in
    # ~1ms without Gemini ever seeing it. Page count cannot prove a side is missing.
    content = _pdf(1)
    result = validate_document_request(
        file=_upload("doc.pdf", "application/pdf", content),
        content=content,
        document_name="Doc",
        document_type=document_type,
        settings=_settings(),
    )
    assert result.page_count == 1
    assert result.document_type == document_type


def test_valid_png_is_accepted() -> None:
    content = _png()
    result = validate_document_request(
        file=_upload("id.png", "image/png", content),
        content=content,
        document_name="National ID",
        document_type="national_id",
        settings=_settings(),
    )
    assert result.mime_type == "image/png"


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        ("national_id", "national_id"),
        ("driving_license", "driving_license"),
        ("Trade License", "trade_license"),
        ("tenancy-contract", "tenancy_contract"),
        ("  Salary Certificate  ", "salary_certificate"),
    ],
)
def test_any_document_type_is_accepted_and_normalized(submitted: str, expected: str) -> None:
    content = _png()
    result = validate_document_request(
        file=_upload("doc.png", "image/png", content),
        content=content,
        document_name="Some Document",
        document_type=submitted,
        settings=_settings(),
    )
    assert result.document_type == expected


def test_non_kyc_document_type_needs_no_prior_national_id_upload() -> None:
    content = _png()
    result = validate_document_request(
        file=_upload("license.png", "image/png", content),
        content=content,
        document_name="Driving License",
        document_type="driving_license",
        settings=_settings(),
    )
    assert result.document_type == "driving_license"


def test_passport_no_longer_requires_a_national_id_first() -> None:
    content = _png()
    result = validate_document_request(
        file=_upload("passport.png", "image/png", content),
        content=content,
        document_name="Passport",
        document_type="passport",
        settings=_settings(),
    )
    assert result.document_type == "passport"


@pytest.mark.parametrize("document_type", ["", "   ", "a" * 51, "bad/type", "type;drop", "emoji_\U0001f600"])
def test_malformed_document_type_is_rejected(document_type: str) -> None:
    content = _png()
    with pytest.raises(HTTPException, match="document type is not valid"):
        validate_document_request(
            file=_upload("doc.png", "image/png", content),
            content=content,
            document_name="Some Document",
            document_type=document_type,
            settings=_settings(),
        )


def test_rate_limit_rejects_after_configured_count() -> None:
    settings = _settings(rate_limit_requests=1, rate_limit_window_seconds=60)
    enforce_rate_limit("rate-test", settings)
    with pytest.raises(HTTPException) as error:
        enforce_rate_limit("rate-test", settings)
    assert error.value.status_code == 429


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("Cheque", "cheque"),
        ("Cheque Copy", "cheque"),
        ("Security-Cheque", "cheque"),
        ("Salary Statement", "salary_certificate"),
        ("Salary Certificate", "salary_certificate"),
        ("Payslip", "salary_certificate"),
        ("Emirates ID", "national_id"),
        ("nationalid", "national_id"),
        ("Utility Bill", "utility_bill"),  # unknown types pass through unchanged
    ],
)
def test_declared_type_is_normalised_to_its_checked_spelling(declared: str, expected: str) -> None:
    validated = validate_document_request(
        file=_upload("doc.png", "image/png", _png()),
        content=_png(),
        document_name="Doc",
        document_type=declared,
        settings=_settings(),
    )
    assert validated.document_type == expected
