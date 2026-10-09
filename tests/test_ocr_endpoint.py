"""Endpoint tests for /ocr: one merged response no matter how many documents were uploaded.

Gemini is stubbed out throughout - extraction and the two verification calls are all faked, so
what is under test is request handling and response shape, never a live model response. Every
problem - file validation, type mismatch, incomplete front/back, self-verification, cross-
verification - is asserted to stop the *whole* request with blank `data` and the reason in
`errorInfo`; a clean set of documents is asserted to merge into one object with every field its
own document supplied, never a per-document breakdown and never a new key.
"""

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.schemas.ocr import DATA_FIELDS
from app.services import ocr_pipeline
from app.services.gemini import get_gemini_service


def _png(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _file(name: str, content: bytes | None = None) -> tuple[str, tuple[str, bytes, str]]:
    return ("file", (name, content if content is not None else _png(), "image/png"))


def _raw(
    types_present: list[str],
    complete: bool = True,
    front_visible: bool | None = None,
    back_visible: bool | None = None,
    **fields: Any,
) -> dict[str, Any]:
    data = dict.fromkeys(DATA_FIELDS)
    data.update(fields)
    raw: dict[str, Any] = {"data": data, "documentComplete": complete, "documentTypesPresent": types_present}
    if front_visible is not None:
        raw["frontVisible"] = front_visible
    if back_visible is not None:
        raw["backVisible"] = back_visible
    return raw


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # TestClient speaks plain http, and detection/verification would each reach the model a second
    # time; all three are switched off at the source rather than worked around per request.
    monkeypatch.setattr(settings, "require_https", False)
    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    return TestClient(app)


@pytest.fixture
def extractions(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every extraction call and answer it as the document type that was asked for."""
    calls: list[dict[str, Any]] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _raw([kwargs["document_type"]], Tenant_Name_En=f"Holder of {kwargs['filename']}")

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    return calls


def _keys(payload: dict[str, Any]) -> set[str]:
    return set(payload)


_ORIGINAL_KEYS = {"success", "userName", "tenantType", "subscriptionId", "ocrReferenceId", "data", "missingInfo", "errorInfo"}


def test_single_document_still_answers_with_one_object(client: TestClient, extractions: list[dict[str, Any]]) -> None:
    response = client.post(
        "/api/v1/ocr",
        files=[_file("passport.png")],
        data={"documentName": "A. Rahman passport", "documentType": "passport"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert _keys(payload) == _ORIGINAL_KEYS
    assert payload["success"] is True
    assert payload["data"]["Tenant_Name_En"] == "Holder of passport.png"
    assert len(extractions) == 1


def test_multiple_documents_merge_into_one_object_with_every_respective_key_filled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        doc_type = kwargs["document_type"]
        if doc_type == "national_id":
            return _raw(["national_id"], Tenant_Name_En="Ahmed", National_Id="784-1", National_Id_Expiry="2030-01-01")
        if doc_type == "passport":
            return _raw(["passport"], Passport_Number="P123", Passport_Expiry="2029-01-01")
        return _raw(["visa"], Visa_Number="V999", Visa_Expiry="2028-01-01")

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id.png"), _file("passport.png"), _file("visa.png")],
        data={
            "documentName": ["ID", "Passport", "Visa"],
            "documentType": ["national_id", "passport", "visa"],
        },
    )

    payload = response.json()
    assert response.status_code == 200
    # Not a list, not a {"documents": [...]} wrapper - the exact same object shape as one document.
    assert isinstance(payload, dict)
    assert _keys(payload) == _ORIGINAL_KEYS
    assert payload["success"] is True
    assert payload["data"]["Tenant_Name_En"] == "Ahmed"
    assert payload["data"]["National_Id"] == "784-1"
    assert payload["data"]["Passport_Number"] == "P123"
    assert payload["data"]["Visa_Number"] == "V999"
    assert payload["errorInfo"] == []


def test_a_type_mismatch_stops_the_whole_request_with_blank_data(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        types = ["passport"] if kwargs["filename"] == "mislabelled.png" else [kwargs["document_type"]]
        return _raw(types, Tenant_Name_En="Holder")

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("visa.png"), _file("mislabelled.png")],
        data={"documentName": ["Visa", "ID"], "documentType": ["visa", "national_id"]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert isinstance(payload, dict)
    assert _keys(payload) == _ORIGINAL_KEYS
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert "does not look like a National ID" in payload["errorInfo"][0]["DocumentErrorToShow"]


def test_incomplete_front_back_document_stops_the_whole_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "detection_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        if kwargs["document_type"] == "national_id":
            return _raw(["national_id"], complete=False, front_visible=True, back_visible=False, Tenant_Name_En="Holder")
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder", Passport_Number="P1")

    detection_calls: list[Any] = []

    async def fake_detection(**kwargs: Any) -> None:
        detection_calls.append(kwargs)

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(ocr_pipeline, "run_detection", fake_detection)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id.png"), _file("passport.png")],
        data={"documentName": ["ID", "Passport"], "documentType": ["national_id", "passport"]},
        headers={"X-Detection": "on"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert "Both sides of the National ID are needed" in payload["errorInfo"][0]["DocumentErrorToShow"]
    # Detection only ever runs once the whole batch has cleared every gate. The National ID's
    # incompleteness stops the request before that point, so the passport - which passed on its
    # own - is never cropped or rotated either.
    assert not detection_calls


def test_front_and_back_uploaded_separately_are_merged_into_one_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)

    extracted_files: list[str] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        extracted_files.append(kwargs["filename"])
        if "front" in kwargs["filename"]:
            return _raw(["national_id"], complete=False, front_visible=True, back_visible=False,
                        Tenant_Name_En="Ahmed", National_Id="784-1")
        return _raw(["national_id"], complete=False, front_visible=False, back_visible=True, Occupation="Engineer")

    pair_calls: list[Any] = []

    async def fake_pair(**kwargs: Any) -> bool:
        pair_calls.append(kwargs)
        return True

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document_pair", fake_pair)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id_front.png"), _file("id_back.png")],
        data={"documentName": ["ID Front", "ID Back"], "documentType": ["national_id", "national_id"]},
        headers={"X-Validation": "on"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["Tenant_Name_En"] == "Ahmed"
    assert payload["data"]["National_Id"] == "784-1"
    assert payload["data"]["Occupation"] == "Engineer"
    assert payload["errorInfo"] == []
    # The two sides' own readings are combined; the merged PDF is never extracted a second time.
    assert sorted(extracted_files) == ["id_back.png", "id_front.png"]
    # Exactly one pairing confirmation, for the one candidate pair - and the two originals collapse
    # into a single merged document, so there is nothing left for cross-verification to compare.
    assert len(pair_calls) == 1


def test_pairing_rejected_by_the_model_stops_the_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        if "front" in kwargs["filename"]:
            return _raw(["national_id"], complete=False, front_visible=True, back_visible=False)
        return _raw(["national_id"], complete=False, front_visible=False, back_visible=True)

    async def fake_pair(**kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document_pair", fake_pair)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id_front.png"), _file("id_back.png")],
        data={"documentName": ["ID Front", "ID Back"], "documentType": ["national_id", "national_id"]},
        headers={"X-Validation": "on"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert "don't look like the front and back of the same card" in payload["errorInfo"][0]["DocumentErrorToShow"]


def test_ambiguous_incomplete_pair_is_rejected_without_a_pairing_call(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        # Both uploads show only the front - there is no back anywhere in the batch to pair either
        # of them with, so this can never be resolved automatically.
        return _raw(["national_id"], complete=False, front_visible=True, back_visible=False)

    pair_calls: list[Any] = []

    async def fake_pair(**kwargs: Any) -> bool:
        pair_calls.append(kwargs)
        return True

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document_pair", fake_pair)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id_a.png"), _file("id_b.png")],
        data={"documentName": ["ID A", "ID B"], "documentType": ["national_id", "national_id"]},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert "couldn't tell which front goes with which back" in payload["errorInfo"][0]["DocumentErrorToShow"]
    assert not pair_calls


def test_a_one_sided_passport_beside_a_merged_national_id_is_complete(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only a National ID needs both sides. A National ID's front and back uploaded separately (a
    # real, resolvable pair) alongside a passport showing a single page must merge the ID and
    # accept the passport as it is - never report it incomplete or as part of an "ambiguous pair".
    monkeypatch.setattr(settings, "validation_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        filename = kwargs["filename"]
        if kwargs["document_type"] == "passport":
            return _raw(["passport"], complete=False, front_visible=True, back_visible=False, Passport_Number="P1")
        if "front" in filename:
            return _raw(["national_id"], complete=False, front_visible=True, back_visible=False, Tenant_Name_En="Ahmed")
        return _raw(["national_id"], complete=False, front_visible=False, back_visible=True)

    async def fake_pair(**kwargs: Any) -> bool:
        return True

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document_pair", fake_pair)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", no_issues)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id_front.png"), _file("id_back.png"), _file("passport_front.png")],
        data={
            "documentName": ["ID Front", "ID Back", "Passport"],
            "documentType": ["national_id", "national_id", "passport"],
        },
        headers={"X-Validation": "on"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["errorInfo"] == []
    assert payload["data"]["Tenant_Name_En"] == "Ahmed"
    assert payload["data"]["Passport_Number"] == "P1"


def test_self_verification_failure_stops_the_whole_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    monkeypatch.setattr(settings, "detection_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder")

    async def fake_verify(**kwargs: Any) -> list[str]:
        return ["The passport expired on yyyy-mm-dd."]

    detection_calls: list[Any] = []

    async def fake_detection(**kwargs: Any) -> None:
        detection_calls.append(kwargs)

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", fake_verify)
    monkeypatch.setattr(ocr_pipeline, "run_detection", fake_detection)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("passport.png")],
        data={"documentName": "Passport", "documentType": "passport"},
        headers={"X-Validation": "on", "X-Detection": "on"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert not detection_calls
    assert any(e["DocumentError"] == "The passport expired on yyyy-mm-dd." for e in payload["errorInfo"])


def test_self_verification_that_cannot_run_fails_the_request_after_one_retry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder")

    attempts: list[Any] = []

    async def failing_verify(**kwargs: Any) -> list[str]:
        attempts.append(kwargs)
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", failing_verify)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("passport.png")],
        data={"documentName": "Passport", "documentType": "passport"},
        headers={"X-Validation": "on"},
    )

    # A document the check could not run on is never passed as if it had been checked: the call
    # is retried once, then the request fails with a plain "try again" message.
    assert len(attempts) == 2
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert "could not be verified right now" in payload["errorInfo"][0]["DocumentErrorToShow"]


def test_self_verification_recovers_when_the_retry_succeeds(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder")

    attempts: list[Any] = []

    async def flaky_verify(**kwargs: Any) -> list[str]:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise RuntimeError("503 busy")
        return []

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", flaky_verify)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("passport.png")],
        data={"documentName": "Passport", "documentType": "passport"},
        headers={"X-Validation": "on"},
    )

    assert len(attempts) == 2
    assert response.json()["success"] is True


def test_cross_verification_conflict_stops_the_whole_request(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder", Date_Of_Birth="1990-01-01")

    cross_calls: list[Any] = []

    async def fake_cross(**kwargs: Any) -> list[str]:
        cross_calls.append(kwargs["documents"])
        return ["'ID' gives a different date of birth than 'Passport'."]

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", fake_cross)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id.png"), _file("passport.png")],
        data={"documentName": ["ID", "Passport"], "documentType": ["national_id", "passport"]},
        headers={"X-Validation": "on"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert len(cross_calls) == 1
    assert {doc["label"] for doc in cross_calls[0]} == {"ID", "Passport"}
    assert payload["success"] is False
    assert all(value is None for value in payload["data"].values())
    assert payload["errorInfo"][0]["DocumentError"] == "'ID' gives a different date of birth than 'Passport'."


def test_detection_runs_for_every_document_only_after_the_batch_clears(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    monkeypatch.setattr(settings, "detection_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder", Date_Of_Birth="1990-01-01")

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    detection_calls: list[Any] = []

    async def fake_detection(**kwargs: Any) -> None:
        detection_calls.append(kwargs)

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", no_issues)
    monkeypatch.setattr(ocr_pipeline, "run_detection", fake_detection)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id.png"), _file("passport.png")],
        data={"documentName": ["ID", "Passport"], "documentType": ["national_id", "passport"]},
        headers={"X-Validation": "on", "X-Detection": "on"},
    )

    assert response.json()["success"] is True
    assert len(detection_calls) == 2


def test_cross_verification_conflict_never_runs_detection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    monkeypatch.setattr(settings, "detection_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder", Date_Of_Birth="1990-01-01")

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    async def fake_cross(**kwargs: Any) -> list[str]:
        return ["'ID' gives a different date of birth than 'Passport'."]

    detection_calls: list[Any] = []

    async def fake_detection(**kwargs: Any) -> None:
        detection_calls.append(kwargs)

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", fake_cross)
    monkeypatch.setattr(ocr_pipeline, "run_detection", fake_detection)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("id.png"), _file("passport.png")],
        data={"documentName": ["ID", "Passport"], "documentType": ["national_id", "passport"]},
        headers={"X-Validation": "on", "X-Detection": "on"},
    )

    payload = response.json()
    assert payload["success"] is False
    # Both documents individually passed self-verification - the old behaviour would have already
    # cropped/rotated each of them by this point. Detection is now deferred past cross-verification,
    # so a conflict found only here still stops it running for either one.
    assert not detection_calls


def test_cross_verification_does_not_run_for_a_single_document(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]])

    cross_calls: list[Any] = []

    async def fake_cross(**kwargs: Any) -> list[str]:
        cross_calls.append(kwargs)
        return []

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", fake_cross)

    client.post(
        "/api/v1/ocr",
        files=[_file("passport.png")],
        data={"documentName": "Passport", "documentType": "passport"},
        headers={"X-Validation": "on"},
    )

    assert not cross_calls


def test_a_bad_file_stops_the_whole_request_with_its_original_status_code(
    client: TestClient, extractions: list[dict[str, Any]]
) -> None:
    response = client.post(
        "/api/v1/ocr",
        files=[_file("good.png"), _file("corrupt.png", b"not a png at all"), _file("also_good.png")],
        data={
            "documentName": ["First", "Second", "Third"],
            "documentType": ["passport", "passport", "visa"],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload["errorInfo"][0]["DocumentFileName"] == "corrupt.png"
    assert all(value is None for value in payload["data"].values())
    # The file before it in upload order was read and extracted; nothing after the bad one was.
    assert [call["filename"] for call in extractions] == ["good.png"]


def test_single_document_rejection_keeps_its_status_code(client: TestClient, extractions: list[dict[str, Any]]) -> None:
    response = client.post(
        "/api/v1/ocr",
        files=[_file("corrupt.png", b"not a png at all")],
        data={"documentName": "Tenant ID", "documentType": "national_id"},
    )

    assert response.status_code == 400
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload["errorInfo"][0]["DocumentFileName"] == "corrupt.png"
    assert not extractions


def test_unpaired_fields_are_rejected_before_any_model_call(
    client: TestClient, extractions: list[dict[str, Any]]
) -> None:
    response = client.post(
        "/api/v1/ocr",
        files=[_file("one.png"), _file("two.png")],
        data={"documentName": ["Only one name"], "documentType": ["passport", "visa"]},
    )

    assert response.status_code == 400
    assert "its own documentName and documentType" in response.json()["errorInfo"][0]["DocumentError"]
    assert not extractions


def test_x_validation_header_off_skips_the_verification_call(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)  # header must override this

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]])

    calls: list[Any] = []

    async def fake_verify(**kwargs: Any) -> list[str]:
        calls.append(kwargs)
        return ["should never appear"]

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", fake_verify)

    response = client.post(
        "/api/v1/ocr",
        files=[_file("passport.png")],
        data={"documentName": "Passport", "documentType": "passport"},
        headers={"X-Validation": "off"},
    )

    assert response.status_code == 200
    assert not calls
    assert response.json()["errorInfo"] == []


def test_identical_file_reuses_extraction_and_passed_self_check(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    monkeypatch.setattr(settings, "result_cache_seconds", 10800)
    calls = {"extract": 0, "verify": 0}

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        calls["extract"] += 1
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder")

    async def clean_verify(**kwargs: Any) -> list[str]:
        calls["verify"] += 1
        return []

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", clean_verify)

    for _ in range(2):
        response = client.post(
            "/api/v1/ocr",
            files=[_file("passport.png")],
            data={"documentName": "Passport", "documentType": "passport"},
            headers={"X-Validation": "on"},
        )
        assert response.json()["data"]["Tenant_Name_En"] == "Holder"

    # The second, byte-identical upload is answered from the cache: no further Gemini calls.
    assert calls == {"extract": 1, "verify": 1}


def test_a_self_check_finding_is_never_reused(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    monkeypatch.setattr(settings, "result_cache_seconds", 10800)
    verify_calls: list[Any] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder")

    async def finding_verify(**kwargs: Any) -> list[str]:
        verify_calls.append(kwargs)
        return ["The passport shows an expiry of 12 May 2023, which is before today"]

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "verify_document", finding_verify)

    for _ in range(2):
        response = client.post(
            "/api/v1/ocr",
            files=[_file("passport.png")],
            data={"documentName": "Passport", "documentType": "passport"},
            headers={"X-Validation": "on"},
        )
        assert response.json()["success"] is False

    # A rejected file is checked afresh every time, so a corrected re-check can never be blocked.
    assert len(verify_calls) == 2


def test_result_cache_can_be_switched_off(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "result_cache_seconds", 0)
    calls: list[Any] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _raw([kwargs["document_type"]], Tenant_Name_En="Holder")

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)

    for _ in range(2):
        client.post(
            "/api/v1/ocr",
            files=[_file("passport.png")],
            data={"documentName": "Passport", "documentType": "passport"},
        )

    assert len(calls) == 2
