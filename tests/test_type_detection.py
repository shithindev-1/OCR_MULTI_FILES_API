"""Endpoint tests for the type-detection (trial) endpoints, /ocr/leasing/detect[/upload].

Gemini is stubbed throughout: `segment_documents` says what each file holds and where, `extract`
answers per file name. Under test: splitting a file into its documents, the two-reading type check,
ignored unsupported documents, the mandatory-document rule on found types, and the response shape.
"""

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.schemas.ocr import COMPANY_DATA_FIELDS, DATA_FIELDS
from app.services import blob_storage
from app.services.gemini import get_gemini_service
from app.services.leasing import sources
from app.services.type_detection import service

UPLOAD_URL = "/api/v1/ocr/leasing/detect/upload"
FULL_PAGE = [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}]
TOP_HALF = [{"x": 0.05, "y": 0.05}, {"x": 0.95, "y": 0.05}, {"x": 0.95, "y": 0.45}, {"x": 0.05, "y": 0.45}]
BOTTOM_HALF = [{"x": 0.05, "y": 0.55}, {"x": 0.95, "y": 0.55}, {"x": 0.95, "y": 0.95}, {"x": 0.05, "y": 0.95}]


# Files are segmented concurrently, so the segmentation stub cannot rely on call order. Each file
# name gets its own image width instead, and the stub answers by the width it is shown.
_WIDTHS: dict[str, int] = {}


def _png(name: str = "") -> bytes:
    width = _WIDTHS.setdefault(name, 400 + len(_WIDTHS))
    buffer = io.BytesIO()
    Image.new("RGB", (width, 300), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _region(document_type: str, number: int, corners: list[dict[str, float]] = FULL_PAGE, side: str = "single") -> dict[str, Any]:
    return {"document_type": document_type, "side": side, "document_number": number, "corners": corners}


# Identity fields that make documents read as the same person.
AHMED = {"Tenant_Name_En": "Ahmed Khan", "Date_Of_Birth": "01/03/1990"}
SARA = {"Tenant_Name_En": "Sara Ali", "Date_Of_Birth": "15/07/1985"}


def _raw(types_present: list[str], fields: list[str] = DATA_FIELDS, **values: Any) -> dict[str, Any]:
    data = dict.fromkeys(fields)
    data.update(values)
    return {"data": data, "documentComplete": True, "documentTypesPresent": types_present,
            "frontVisible": True, "backVisible": True}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: None)
    return TestClient(app)


def _stub(monkeypatch: pytest.MonkeyPatch, *, segments: dict[str, list[dict[str, Any]]],
          answers: dict[str, dict[str, Any]]) -> list[str]:
    """`segments`: the regions found in each (single-page) file, by file name. `answers`: extraction
    answer by the start of the file name it is read from. Returns the file names extraction was
    called with."""
    by_width = {_WIDTHS.setdefault(name, 400 + len(_WIDTHS)): regions for name, regions in segments.items()}
    extracted: list[str] = []

    async def fake_segment(page_images_png: list[bytes]) -> list[dict[str, Any]]:
        with Image.open(io.BytesIO(page_images_png[0])) as image:
            return [{"regions": by_width[image.width]}]

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        extracted.append(kwargs["filename"])
        return next(answer for prefix, answer in answers.items() if kwargs["filename"].startswith(prefix))

    monkeypatch.setattr(get_gemini_service(), "segment_documents", fake_segment)
    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    return extracted


def _upload(client: TestClient, names: list[str], tenant_type: str = "individual", **extra: Any) -> Any:
    files = [("file", (name, _png(name), "image/png")) for name in names]
    data = {"tenant_type": tenant_type, "document_refnumber": "ref-1", **extra}
    return client.post(UPLOAD_URL, files=files, data=data)


def test_single_document_file_is_read_as_is_and_typed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _stub(
        monkeypatch,
        segments={"eid.png": [_region("national_id", 1)]},
        answers={"eid.png": _raw(["national_id"], National_Id="784-1985-1234567-1")},
    )

    body = _upload(client, ["eid.png"]).json()

    assert body["status"] == "success"
    # One document in the file: the original is read, never a cut-out.
    assert extracted == ["eid.png"]
    [tenant] = body["form_data"]
    assert (tenant["tenant_id"], tenant["status"], tenant["national_id"]) == ("tenant_1", "success", "784-1985-1234567-1")
    assert body["missing"][0]["tenant_id"] == "tenant_1"
    assert "national_id" not in body["missing"][0]["fields"]
    [document] = body["documents"]
    assert document == {
        "tenant_ids": ["tenant_1"],
        "document_type": "national_id",
        "status": "valid",
        "merged": False,
        "sources": [{"source": "eid.png", "file_name": "eid.png", "pages": [1], "ocr_file": None}],
        "error": None,
    }


def test_file_with_two_documents_is_split_and_each_checked(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _stub(
        monkeypatch,
        segments={"scan.png": [_region("national_id", 1, TOP_HALF), _region("passport", 2, BOTTOM_HALF)]},
        answers={
            "scan_1_national_id": _raw(["national_id"], National_Id="784-1985-1234567-1", **AHMED),
            "scan_2_passport": _raw(["passport"], Passport_Number="N1234567", **AHMED),
        },
    )

    body = _upload(client, ["scan.png"]).json()

    assert body["status"] == "success"
    assert sorted(extracted) == ["scan_1_national_id.png", "scan_2_passport.png"]
    [tenant] = body["form_data"]
    assert (tenant["national_id"], tenant["passport_number"]) == ("784-1985-1234567-1", "N1234567")
    assert [(d["document_type"], d["status"]) for d in body["documents"]] == [
        ("national_id", "valid"),
        ("passport", "valid"),
    ]


def test_type_must_be_confirmed_by_the_extraction_reading(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(
        monkeypatch,
        segments={"card.png": [_region("national_id", 1)], "pp.png": [_region("passport", 1)]},
        answers={"card.png": _raw(["visa"]), "pp.png": _raw([])},
    )

    body = _upload(client, ["card.png", "pp.png"]).json()

    assert body["status"] == "failed"
    card, passport = body["documents"]
    assert card["status"] == "invalid"
    assert card["error"] == "We couldn't clearly identify 'National ID'. Please upload a clearer copy."
    # An empty second reading is not agreement.
    assert passport["status"] == "invalid"
    assert passport["error"] == "We couldn't clearly identify 'Passport'. Please upload a clearer copy."
    assert body["form_data"] == []


def test_unsupported_documents_are_ignored(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    extracted = _stub(
        monkeypatch,
        segments={"eid.png": [_region("national_id", 1)], "bill.png": [_region("other", 1)]},
        answers={"eid.png": _raw(["national_id"])},
    )

    body = _upload(client, ["eid.png", "bill.png"]).json()

    assert body["status"] == "success"
    assert extracted == ["eid.png"]
    ignored = body["documents"][-1]
    assert (ignored["document_type"], ignored["status"], ignored["error"]) == ("other", "ignored", None)
    assert ignored["sources"][0]["file_name"] == "bill.png"


def test_missing_national_id_is_rejected_for_an_individual(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, segments={"pp.png": [_region("passport", 1)], "bill.png": [_region("other", 1)]}, answers={})

    response = _upload(client, ["pp.png", "bill.png"])

    assert response.status_code == 400
    assert response.json()["detail"].startswith("A National ID is required for an individual.")


def test_missing_trade_licence_is_rejected_for_a_company(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, segments={"eid.png": [_region("national_id", 1)]}, answers={})

    response = _upload(client, ["eid.png"], tenant_type="company")

    assert response.status_code == 400
    assert response.json()["detail"].startswith("A trade licence or initial approval is required for a company.")


def test_company_with_a_listed_signatory_succeeds(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    company_fields = COMPANY_DATA_FIELDS + ["License_Members", "Holder_Name"]
    _stub(
        monkeypatch,
        segments={"tl.png": [_region("trade_license", 1)], "eid.png": [_region("national_id", 1)]},
        answers={
            "tl.png": _raw(["trade_license"], company_fields, Company_Name_En="Alpha Trading LLC",
                           License_Members="Ahmed Khan (Manager)"),
            "eid.png": _raw(["national_id"], company_fields, Signatory_Name_En="Ahmed Khan"),
        },
    )

    body = _upload(client, ["tl.png", "eid.png"], tenant_type="company").json()

    assert body["status"] == "success"
    [tenant] = body["form_data"]
    assert (tenant["company_name_en"], tenant["signatory_name_en"]) == ("Alpha Trading LLC", "Ahmed Khan")


def test_company_whose_listed_people_were_not_uploaded_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The person uploaded is not on the licence: the person fails, and so does the licence's company
    # tenant - it never passes with company details alone.
    company_fields = COMPANY_DATA_FIELDS + ["License_Members", "Holder_Name"]
    _stub(
        monkeypatch,
        segments={"tl.png": [_region("trade_license", 1)], "eid.png": [_region("national_id", 1)]},
        answers={
            "tl.png": _raw(["trade_license"], company_fields, Company_Name_En="Alpha Trading LLC",
                           License_Members="Omar Saleh (Partner)"),
            "eid.png": _raw(["national_id"], company_fields, Signatory_Name_En="Ahmed Khan"),
        },
    )

    body = _upload(client, ["tl.png", "eid.png"], tenant_type="company").json()

    assert body["status"] == "failed"
    [tenant] = body["form_data"]
    assert (tenant["status"], tenant["company_name_en"]) == ("failed", None)
    licence, person = body["documents"]
    assert licence["error"] == (
        "None of the people listed on this trade licence or initial approval were uploaded. "
        "Please upload the ID documents of a person it lists."
    )
    assert person["error"].startswith(
        "The person on 'National ID' is not listed on any uploaded trade licence or initial approval."
    )


def test_expired_document_is_rejected_with_its_own_message(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    _stub(monkeypatch, segments={"eid.png": [_region("national_id", 1)]}, answers={"eid.png": _raw(["national_id"])})
    message = "The National ID expired on 12/05/2025, before today. Please upload a valid National ID."

    async def fake_verify(**kwargs: Any) -> list[str]:
        assert kwargs["document_type"] == "national_id"
        return [message]

    monkeypatch.setattr(get_gemini_service(), "verify_document", fake_verify)

    body = _upload(client, ["eid.png"]).json()

    assert body["status"] == "failed"
    [document] = body["documents"]
    assert (document["status"], document["error"]) == ("invalid", message)


@pytest.mark.parametrize("field", ["document_type", "document_name", "document_id"])
def test_declared_fields_are_rejected(client: TestClient, field: str) -> None:
    response = _upload(client, ["eid.png"], **{field: "passport"})

    assert response.status_code == 400
    assert field in response.json()["detail"]


def test_json_route_rejects_declared_fields(client: TestClient) -> None:
    response = client.post("/api/v1/ocr/leasing/detect", json={
        "tenant_type": "individual", "document_refnumber": "ref-1",
        "sources": [{"source": "https://example.com/a.png", "document_type": "passport"}],
    })

    assert response.status_code == 422


def test_unreadable_file_fails_without_checking_the_others(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, segments={"eid.png": [_region("national_id", 1)]}, answers={})

    files = [("file", ("eid.png", _png("eid.png"), "image/png")), ("file", ("broken.png", b"not an image", "image/png"))]
    body = client.post(UPLOAD_URL, files=files, data={"tenant_type": "individual", "document_refnumber": "ref-1"}).json()

    assert body["status"] == "failed"
    eid, broken = body["documents"]
    assert eid["status"] == "not_checked"
    assert broken["status"] == "invalid"
    assert broken["error"].startswith("'broken.png':")


def test_front_and_back_of_one_card_become_one_document_and_others_are_cut_out() -> None:
    import numpy as np

    from app.services.type_detection.segmentation import build_segments

    page = np.full((300, 400, 3), 255, dtype=np.uint8)
    segments = build_segments(
        [page, page],
        [
            {"regions": [_region("national_id", 1, TOP_HALF, "front"), _region("passport", 2, BOTTOM_HALF)]},
            {"regions": [_region("national_id", 1, TOP_HALF, "back")]},
        ],
        settings,
    )

    card, passport = segments
    assert (card.document_type, card.pages, card.extension) == ("national_id", [1, 2], ".pdf")
    assert card.content.startswith(b"%PDF-")
    assert (passport.document_type, passport.pages, passport.extension) == ("passport", [1], ".png")


def test_each_cut_out_document_gets_its_own_blob_result_path() -> None:
    from app.services.type_detection.segmentation import Segment

    item = {"source": "s", "filename": "Scan.pdf", "content": b"", "content_type": "application/pdf",
            "blob_path": "ApplicationFiles/42/Scan.pdf"}
    cut = service._batch_item(item, Segment("passport", [2], b"x", ".jpg"), "Passport - Scan.pdf", 3)
    whole = service._batch_item(item, Segment("passport", [1]), "Passport - Scan.pdf", 1)

    assert cut["blob_path"] == "ApplicationFiles/42/Scan_3_passport.jpg"
    assert cut["content_type"] == "image/jpeg"
    # A file holding one document keeps today's naming: <name>_ocr.<ext> beside the original.
    assert whole["blob_path"] == "ApplicationFiles/42/Scan.pdf"


# --- background jobs: POST /detect/jobs[/upload], polled with the normal GET /jobs/{job_id} -------

JOBS_URL = "/api/v1/ocr/leasing/jobs"


@pytest.fixture
def live_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A client whose event loop stays alive between requests, so a background job keeps running."""
    import app.main as main_module

    async def no_warm_up() -> None:
        return None

    async def fake_download(client: Any, source: Any) -> tuple[bytes, str, str]:
        return _png(source.document_name), source.document_name, "image/png"

    monkeypatch.setattr(main_module, "_warm_up_connections", no_warm_up)
    monkeypatch.setattr(settings, "output_retention_hours", 0)
    monkeypatch.setattr(settings, "require_https", False)
    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: None)
    monkeypatch.setattr(sources, "download_source", fake_download)
    with TestClient(app) as live:
        yield live


def _finished_job(client: TestClient, job_id: str) -> dict[str, Any]:
    import time

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"{JOBS_URL}/{job_id}").json()
        if job["status"] == "Completed":
            return job
        time.sleep(0.05)
    raise AssertionError("job never completed")


def test_json_job_returns_a_job_id_then_the_result_on_the_normal_jobs_endpoint(
    live_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(
        monkeypatch,
        segments={"eid.png": [_region("national_id", 1)], "pp.png": [_region("passport", 1)]},
        answers={"eid.png": _raw(["national_id"], **AHMED), "pp.png": _raw(["passport"], Passport_Number="N1234567", **AHMED)},
    )

    started = live_client.post("/api/v1/ocr/leasing/detect/jobs", json={
        "tenant_type": "individual", "document_refnumber": "job-1",
        "sources": [{"source": "https://files.example.com/eid.png"}, {"source": "https://files.example.com/pp.png"}],
    })

    assert started.status_code == 202
    job = _finished_job(live_client, started.json()["job_id"])
    assert job["percentage"] == 100
    assert job["result"]["status"] == "success"
    assert job["result"]["form_data"][0]["passport_number"] == "N1234567"
    assert [d["document_type"] for d in job["result"]["documents"]] == ["national_id", "passport"]


def test_job_without_the_mandatory_document_ends_with_that_message(
    live_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, segments={"pp.png": [_region("passport", 1)]}, answers={})

    started = live_client.post("/api/v1/ocr/leasing/detect/jobs", json={
        "tenant_type": "individual", "document_refnumber": "job-2",
        "sources": [{"source": "https://files.example.com/pp.png"}],
    })

    job = _finished_job(live_client, started.json()["job_id"])
    assert job["result"]["status"] == "failed"
    [document] = job["result"]["documents"]
    assert document["error"].startswith("A National ID is required for an individual.")
    assert document["sources"][0]["file_name"] == "pp.png"


@pytest.mark.parametrize("url", ["/api/v1/ocr/leasing/detect/jobs", "/api/v1/ocr/leasing/detect"])
def test_an_empty_source_is_rejected_before_anything_starts(live_client: TestClient, url: str) -> None:
    response = live_client.post(url, json={
        "tenant_type": "company", "document_refnumber": "job-4", "sources": [{"source": ""}],
    })

    assert response.status_code == 422
    assert response.json() == {"detail": "Source 1 is empty. Please add a file link."}


def test_no_sources_is_rejected_before_a_job_starts(live_client: TestClient) -> None:
    response = live_client.post("/api/v1/ocr/leasing/detect/jobs", json={
        "tenant_type": "company", "document_refnumber": "job-5", "sources": [],
    })

    assert response.status_code == 400
    assert response.json() == {"detail": "Please add at least one document."}


def test_upload_job_rejects_declared_fields_before_it_starts(live_client: TestClient) -> None:
    response = live_client.post(
        "/api/v1/ocr/leasing/detect/jobs/upload",
        files=[("file", ("eid.png", _png("eid.png"), "image/png"))],
        data={"tenant_type": "individual", "document_refnumber": "job-3", "document_type": "national_id"},
    )

    assert response.status_code == 400


def test_short_messages_are_asked_for_and_one_conflict_is_reported_once(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "validation_enabled", True)
    _stub(
        monkeypatch,
        segments={"eid.png": [_region("national_id", 1)], "pp.png": [_region("passport", 1)]},
        answers={"eid.png": _raw(["national_id"], **AHMED), "pp.png": _raw(["passport"], **AHMED)},
    )
    conflict = "'National ID' and 'Passport' belong to different people. Please upload documents of the same person."
    asked: list[bool] = []

    async def fake_verify(**kwargs: Any) -> list[str]:
        asked.append(kwargs.get("short_messages", False))
        return []

    async def fake_cross(**kwargs: Any) -> list[str]:
        asked.append(kwargs.get("short_messages", False))
        return [conflict, conflict]

    monkeypatch.setattr(get_gemini_service(), "verify_document", fake_verify)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", fake_cross)

    body = _upload(client, ["eid.png", "pp.png"]).json()

    assert asked == [True, True, True]
    assert body["status"] == "failed"
    assert [document["error"] for document in body["documents"]] == [conflict, conflict]


# --- several tenants in one request ----------------------------------------------------------------


def test_two_people_become_two_tenants_with_their_own_form_data(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(
        monkeypatch,
        segments={
            "a_eid.png": [_region("national_id", 1)], "a_pp.png": [_region("passport", 1)],
            "s_eid.png": [_region("national_id", 1)],
        },
        answers={
            "a_eid.png": _raw(["national_id"], National_Id="784-1990-1111111-1", **AHMED),
            "a_pp.png": _raw(["passport"], Passport_Number="P1111111", **AHMED),
            "s_eid.png": _raw(["national_id"], National_Id="784-1985-2222222-2", **SARA),
        },
    )

    body = _upload(client, ["a_eid.png", "a_pp.png", "s_eid.png"]).json()

    assert body["status"] == "success"
    ahmed, sara = body["form_data"]
    assert (ahmed["tenant_id"], ahmed["tenant_name_en"], ahmed["passport_number"]) == ("tenant_1", "Ahmed Khan", "P1111111")
    assert (sara["tenant_id"], sara["tenant_name_en"], sara["passport_number"]) == ("tenant_2", "Sara Ali", None)
    assert [entry["tenant_id"] for entry in body["missing"]] == ["tenant_1", "tenant_2"]
    assert "passport_number" in body["missing"][1]["fields"]
    assert [d["tenant_ids"] for d in body["documents"]] == [["tenant_1"], ["tenant_1"], ["tenant_2"]]


def test_a_tenant_without_a_national_id_fails_alone(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(
        monkeypatch,
        segments={"a_eid.png": [_region("national_id", 1)], "s_pp.png": [_region("passport", 1)]},
        answers={
            "a_eid.png": _raw(["national_id"], **AHMED),
            "s_pp.png": _raw(["passport"], Passport_Number="P2222222", **SARA),
        },
    )

    body = _upload(client, ["a_eid.png", "s_pp.png"]).json()

    assert body["status"] == "failed"
    ahmed, sara = body["form_data"]
    assert (ahmed["status"], ahmed["tenant_name_en"]) == ("success", "Ahmed Khan")
    assert (sara["status"], sara["passport_number"]) == ("failed", None)
    passport = body["documents"][1]
    assert passport["error"] == (
        "No National ID was uploaded for the person on 'Passport'. Please upload their National ID."
    )


def test_documents_linked_by_an_id_number_are_one_person(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(
        monkeypatch,
        segments={"eid.png": [_region("national_id", 1)], "visa.png": [_region("visa", 1)]},
        answers={
            # The visa spells the name differently, but prints the same ID number.
            "eid.png": _raw(["national_id"], National_Id="784-1990-1111111-1", Tenant_Name_En="Mohammed Ali"),
            "visa.png": _raw(["visa"], National_Id="784199011111111", Tenant_Name_En="Mohamad Aly", Visa_Number="201/2024/1"),
        },
    )

    body = _upload(client, ["eid.png", "visa.png"]).json()

    assert body["status"] == "success"
    [tenant] = body["form_data"]
    assert tenant["visa_number"] == "201/2024/1"


def test_company_tenants_are_one_per_licence_and_listed_signatory(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    company_fields = COMPANY_DATA_FIELDS + ["License_Members", "Holder_Name"]
    _stub(
        monkeypatch,
        segments={
            "tl_alpha.png": [_region("trade_license", 1)], "tl_beta.png": [_region("trade_license", 1)],
            "eid_ahmed.png": [_region("national_id", 1)], "eid_omar.png": [_region("national_id", 1)],
            "eid_zaid.png": [_region("national_id", 1)],
        },
        answers={
            "tl_alpha.png": _raw(["trade_license"], company_fields, Company_Name_En="Alpha Trading LLC",
                                 Trade_License_Number="1001", License_Members="Ahmed Khan (Manager)"),
            "tl_beta.png": _raw(["trade_license"], company_fields, Company_Name_En="Beta Foods LLC",
                                Trade_License_Number="2002", License_Members="Omar Saleh (Partner)"),
            "eid_ahmed.png": _raw(["national_id"], company_fields, Signatory_Name_En="Ahmed Khan", Signatory_National_Id="784-11111"),
            "eid_omar.png": _raw(["national_id"], company_fields, Signatory_Name_En="Omar Saleh", Signatory_National_Id="784-22222"),
            "eid_zaid.png": _raw(["national_id"], company_fields, Signatory_Name_En="Zaid Noor", Signatory_National_Id="784-33333"),
        },
    )

    body = _upload(client, ["tl_alpha.png", "tl_beta.png", "eid_ahmed.png", "eid_omar.png", "eid_zaid.png"],
                   tenant_type="company").json()

    alpha, beta = body["form_data"]
    assert (alpha["company_name_en"], alpha["signatory_name_en"], alpha["signatory_designation"]) == (
        "Alpha Trading LLC", "Ahmed Khan", "Manager",
    )
    assert (beta["company_name_en"], beta["signatory_name_en"]) == ("Beta Foods LLC", "Omar Saleh")
    assert alpha["status"] == beta["status"] == "success"
    # Zaid is on no uploaded licence: the request fails, on his document only.
    assert body["status"] == "failed"
    zaid = body["documents"][-1]
    assert zaid["tenant_ids"] == []
    assert zaid["error"].startswith(
        "The person on 'National ID 3' is not listed on any uploaded trade licence or initial approval."
    )


def test_processed_files_are_named_after_their_type(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.leasing.outputs import assign_type_file_names

    items = [
        {"blob_path": "Apps/1/scan_1_national_id.jpg", "source_blob_path": "Apps/1/scan.pdf"},
        {"blob_path": "Apps/1/scan_2_passport.jpg", "source_blob_path": "Apps/1/scan.pdf"},
        {"blob_path": "Apps/1/National_ID.pdf"},  # an original already called National_ID
        {"blob_path": "Apps/2/eid.pdf"},
    ]
    documents = {
        (0,): {"document_type": "national_id"}, (1,): {"document_type": "passport"},
        (2,): {"document_type": "national_id"}, (3,): {"document_type": "national_id"},
    }

    assign_type_file_names(documents, items)

    assert [documents[key]["_ocr_name"] for key in sorted(documents)] == [
        # The original named National_ID is never overwritten.
        "National_ID_ocr", "Passport", "National_ID_2", "National_ID",
    ]


def test_processed_files_carry_their_tenant_when_there_are_several() -> None:
    from app.services.leasing.outputs import assign_type_file_names

    items = [{"blob_path": f"Apps/1/f{index}.pdf"} for index in range(4)]
    documents = {
        (0,): {"document_type": "national_id"}, (1,): {"document_type": "passport"},
        (2,): {"document_type": "national_id"}, (3,): {"document_type": "trade_license"},
    }

    assign_type_file_names(documents, items, {(0,): [1], (1,): [1], (2,): [2], (3,): [1, 2]})

    assert [documents[key]["_ocr_name"] for key in sorted(documents)] == [
        "Tenant_1_National_ID", "Tenant_1_Passport", "Tenant_2_National_ID", "Tenant_1_2_Trade_Licence",
    ]



# --- several people's National ID fronts and backs, uploaded as separate files ---------------------


def _side(front: bool, **values: Any) -> dict[str, Any]:
    raw = _raw(["national_id"], **values)
    raw.update(documentComplete=False, frontVisible=front, backVisible=not front)
    return raw


FOUR_SIDES = ["hus_front.png", "hus_back.png", "wife_front.png", "wife_back.png"]


def _four_sides(monkeypatch: pytest.MonkeyPatch, backs: dict[str, Any] | None = None) -> None:
    backs = backs or {}
    _stub(
        monkeypatch,
        segments={name: [_region("national_id", 1, side="front" if "front" in name else "back")] for name in FOUR_SIDES},
        answers={
            "hus_front.png": _side(True, National_Id="784-1980-1111111-1", **AHMED),
            "hus_back.png": _side(False, **backs.get("hus", {})),
            "wife_front.png": _side(True, National_Id="784-1985-2222222-2", **SARA),
            "wife_back.png": _side(False, **backs.get("wife", {})),
        },
    )


def test_husband_and_wife_cards_are_matched_by_id_number(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _four_sides(monkeypatch, backs={
        "hus": {"National_Id": "784198011111111", "Occupation": "Engineer"},
        "wife": {"National_Id": "784198522222222", "Occupation": "Teacher"},
    })

    body = _upload(client, FOUR_SIDES).json()

    assert body["status"] == "success"
    hus, wife = body["form_data"]
    assert (hus["tenant_name_en"], hus["occupation"]) == ("Ahmed Khan", "Engineer")
    assert (wife["tenant_name_en"], wife["occupation"]) == ("Sara Ali", "Teacher")
    assert [(d["merged"], d["tenant_ids"]) for d in body["documents"]] == [(True, ["tenant_1"]), (True, ["tenant_2"])]


def test_husband_and_wife_cards_are_matched_by_the_same_card_check(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backs carry no readable ID number, so each front/back combination is checked once.
    monkeypatch.setattr(settings, "validation_enabled", True)
    _four_sides(monkeypatch, backs={"hus": {"Occupation": "Engineer"}, "wife": {"Occupation": "Teacher"}})
    widths = {_WIDTHS[name]: name for name in FOUR_SIDES}
    asked: list[tuple[str, str]] = []

    def name_of(content: bytes) -> str:
        with Image.open(io.BytesIO(content)) as image:
            return widths[image.width]

    async def fake_pair(**kwargs: Any) -> bool:
        first, second = name_of(kwargs["first_content"]), name_of(kwargs["second_content"])
        asked.append((first, second))
        return first.split("_")[0] == second.split("_")[0]

    async def no_issues(**kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(get_gemini_service(), "verify_document_pair", fake_pair)
    monkeypatch.setattr(get_gemini_service(), "verify_document", no_issues)
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", no_issues)

    body = _upload(client, FOUR_SIDES).json()

    assert body["status"] == "success"
    assert [entry["occupation"] for entry in body["form_data"]] == ["Engineer", "Teacher"]
    # Four combinations, and no second check when the confirmed pairs are merged.
    assert len(asked) == 4


def test_sides_that_cannot_be_matched_still_get_the_pairing_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Validation off and no ID numbers on the backs: nothing to match two fronts and two backs by.
    _four_sides(monkeypatch)

    body = _upload(client, FOUR_SIDES).json()

    assert body["status"] == "failed"
    assert all("couldn't tell which front goes with which back" in d["error"] for d in body["documents"])


# --- processed files: cropped/rotated at the end, one PDF per document, named after its type -------

CONTAINER = "https://acct.blob.core.windows.net/files-uat"
FOLDER = "ApplicationFiles/LeasingRequestOCR/1/app-1"


class _FakeBlob(blob_storage.BlobStorage):
    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.container_url = CONTAINER
        self.blobs = blobs
        self.uploads: dict[str, tuple[bytes, str]] = {}

    async def download(self, path: str, *, max_bytes: int) -> tuple[bytes, str | None]:
        return self.blobs[path], "application/octet-stream"

    async def upload(self, path: str, content: bytes, content_type: str) -> None:
        self.uploads[path] = (content, content_type)


@pytest.fixture
def cropping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Crop/rotate on, stubbed: one upright crop per region the cut-out holds."""
    from app.services.leasing import batch

    monkeypatch.setattr(settings, "detection_enabled", True)

    async def fake_detection(**kwargs: Any) -> list[Path]:
        folder = Path(kwargs["settings"].detection_output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        pages = 2 if kwargs["mime_type"] == "application/pdf" else 1
        paths = []
        for page in range(1, pages + 1):
            path = folder / f"crop_{kwargs['sha256'][:8]}_{page}.png"
            Image.new("RGB", (80, 50), "gray").save(path, format="PNG")
            paths.append(path)
        return paths

    monkeypatch.setattr(batch, "run_detection", fake_detection)


def test_each_document_of_one_file_is_uploaded_as_its_own_pdf(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, cropping: None, tmp_path: Path,
) -> None:
    storage = _FakeBlob({f"{FOLDER}/scan.png": _png("scan.png")})
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: storage)
    _stub(
        monkeypatch,
        segments={"scan.png": [_region("national_id", 1, TOP_HALF), _region("passport", 2, BOTTOM_HALF)]},
        answers={
            "scan_1_national_id": _raw(["national_id"], National_Id="784-1971-6060254-2", **AHMED),
            "scan_2_passport": _raw(["passport"], Passport_Number="A28334482", **AHMED),
        },
    )

    body = client.post("/api/v1/ocr/leasing/detect", json={
        "tenant_type": "individual", "document_refnumber": "app-1", "sources": [{"source": f"{FOLDER}/scan.png"}],
    }).json()

    assert body["status"] == "success"
    # One file in, two separate processed files out - each a PDF named after its type.
    assert sorted(storage.uploads) == [f"{FOLDER}/Tenant_1_National_ID.pdf", f"{FOLDER}/Tenant_1_Passport.pdf"]
    assert all(content.startswith(b"%PDF-") and kind == "application/pdf" for content, kind in storage.uploads.values())
    assert [d["sources"][0]["ocr_file"] for d in body["documents"]] == [
        f"{CONTAINER}/{FOLDER}/Tenant_1_National_ID.pdf", f"{CONTAINER}/{FOLDER}/Tenant_1_Passport.pdf",
    ]
    # A copy of each is also kept on the server, in the request's own folder.
    assert sorted(path.name for path in (tmp_path / "app-1").glob("*.pdf")) == ["Tenant_1_National_ID.pdf", "Tenant_1_Passport.pdf"]


def test_an_uploaded_file_gets_a_local_pdf_as_its_ocr_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, cropping: None,
) -> None:
    _stub(monkeypatch, segments={"eid.png": [_region("national_id", 1)]}, answers={"eid.png": _raw(["national_id"])})

    body = _upload(client, ["eid.png"]).json()

    ocr_file = Path(body["documents"][0]["sources"][0]["ocr_file"])
    assert ocr_file.name == "Tenant_1_National_ID.pdf"
    assert ocr_file.read_bytes().startswith(b"%PDF-")


def test_several_tenants_get_tenant_named_pdfs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, cropping: None,
) -> None:
    _stub(
        monkeypatch,
        segments={"a.png": [_region("national_id", 1)], "s.png": [_region("national_id", 1)]},
        answers={
            "a.png": _raw(["national_id"], National_Id="784-1990-1111111-1", **AHMED),
            "s.png": _raw(["national_id"], National_Id="784-1985-2222222-2", **SARA),
        },
    )

    body = _upload(client, ["a.png", "s.png"]).json()

    assert [Path(d["sources"][0]["ocr_file"]).name for d in body["documents"]] == [
        "Tenant_1_National_ID.pdf", "Tenant_2_National_ID.pdf",
    ]


# --- the split call's box formats ----------------------------------------------------------------


def _region_box(document_type: str, number: int, box_2d: list[int], side: str = "single") -> dict[str, Any]:
    return {"document_type": document_type, "side": side, "document_number": number, "box_2d": box_2d}


def test_boxes_in_gemini_format_split_a_photo_of_several_cards() -> None:
    # The layout of a real test photo: passport left, ID front and back right, residence card below.
    import numpy as np

    from app.services.type_detection.segmentation import build_segments

    page = np.full((1280, 1055, 3), 255, dtype=np.uint8)
    segments = build_segments([page], [{"regions": [
        _region_box("passport", 1, [0, 11, 642, 576]),
        _region_box("national_id", 2, [24, 638, 214, 997], "front"),
        _region_box("national_id", 2, [257, 638, 444, 999], "back"),
        _region_box("visa", 3, [655, 0, 995, 590]),
    ]}], settings)

    assert [(s.document_type, len(s.regions), s.extension) for s in segments] == [
        ("passport", 1, ".png"), ("national_id", 2, ".pdf"), ("visa", 1, ".png"),
    ]
    # Each cut is its own region, not the whole photo.
    assert all(region.method != "fallback_invalid_response" for s in segments for region in s.regions)


def test_corners_on_a_0_to_1000_scale_are_read_as_such() -> None:
    from app.services.type_detection.segmentation import _fractions

    corners = [{"x": 639, "y": 27}, {"x": 997, "y": 27}, {"x": 997, "y": 213}, {"x": 639, "y": 213}]
    assert _fractions(corners)[0] == {"x": 0.639, "y": 0.027}
    fractions = [{"x": 0.1, "y": 0.2}] * 4
    assert _fractions(fractions) == fractions


# --- every supporting document must be shown to belong to a tenant ------------------------------

MAGDY = {"Tenant_Name_En": "Magdy Abdelrehim Ibrahim Ali", "Date_Of_Birth": "04/05/1971"}


def _magdy_with(monkeypatch: pytest.MonkeyPatch, extra_name: str, extra_type: str, extra: dict[str, Any]) -> None:
    """One photo with Magdy's passport, ID front and back, and residence visa - plus one extra file."""
    _stub(
        monkeypatch,
        segments={
            "photo.png": [
                _region("passport", 1, TOP_HALF),
                _region("national_id", 2, [{"x": 0.6, "y": 0.02}, {"x": 0.98, "y": 0.02}, {"x": 0.98, "y": 0.2}, {"x": 0.6, "y": 0.2}], "front"),
                _region("national_id", 2, [{"x": 0.6, "y": 0.25}, {"x": 0.98, "y": 0.25}, {"x": 0.98, "y": 0.45}, {"x": 0.6, "y": 0.45}], "back"),
                _region("visa", 3, BOTTOM_HALF),
            ],
            extra_name: [_region(extra_type, 1)],
        },
        answers={
            "photo_1_passport": _raw(["passport"], Passport_Number="A28334482", **MAGDY),
            "photo_2_national_id": _raw(["national_id"], National_Id="784-1971-6060254-2", **MAGDY),
            "photo_3_visa": _raw(["visa"], Visa_Number="201/2023/7/434798", Passport_Number="A28334482", **MAGDY),
            extra_name: _raw([extra_type], DATA_FIELDS + ["License_Members"], **extra),
        },
    )


def test_a_trade_licence_that_does_not_list_the_tenant_is_rejected_on_its_own(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reported case: the person's own documents are fine; the licence belongs to someone else.
    _magdy_with(monkeypatch, "licence.png", "trade_license", {
        "License_Members": "Bhargava Subhash Kalluri (Manager)", "Resident_City": "Dubai",
    })

    body = _upload(client, ["photo.png", "licence.png"]).json()

    [tenant] = body["form_data"]
    assert tenant["status"] == "success"
    assert (tenant["tenant_name_en"], tenant["passport_number"]) == ("Magdy Abdelrehim Ibrahim Ali", "A28334482")
    # Nothing is taken from the licence.
    assert tenant["resident_city"] is None
    by_type = {d["document_type"]: d for d in body["documents"]}
    assert [by_type[t]["status"] for t in ("passport", "national_id", "visa")] == ["valid", "valid", "valid"]
    licence = by_type["trade_license"]
    assert (licence["status"], licence["tenant_ids"]) == ("invalid", [])
    assert licence["error"] == (
        "'Trade licence' doesn't list the tenant. Please upload a trade licence that lists the tenant, or remove it."
    )
    # The request as a whole still says something needs fixing.
    assert body["status"] == "failed"


def test_a_trade_licence_that_lists_the_tenant_is_accepted_but_adds_no_data(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _magdy_with(monkeypatch, "licence.png", "trade_license", {
        "License_Members": "MAGDY ABDELREHIM IBRAHIM ALI (Owner)", "Resident_City": "Dubai",
    })

    body = _upload(client, ["photo.png", "licence.png"]).json()

    assert body["status"] == "success"
    licence = next(d for d in body["documents"] if d["document_type"] == "trade_license")
    assert (licence["status"], licence["tenant_ids"]) == ("valid", ["tenant_1"])
    assert body["form_data"][0]["resident_city"] is None


@pytest.mark.parametrize(("extra", "expected"), [
    ({"Tenant_Name_En": "Magdy Abdelrehim Ibrahim Ali"}, None),
    ({"National_Id": "784197160602542"}, None),
    ({"Tenant_Name_En": "Sara Ali"},
     "'Ejari certificate' doesn't mention the tenant. Please upload the tenant's own Ejari certificate, or remove it."),
    ({}, "We couldn't confirm that 'Ejari certificate' belongs to the tenant. Please upload a clearer copy."),
])
def test_a_supporting_document_must_name_the_tenant(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any], expected: str | None,
) -> None:
    _magdy_with(monkeypatch, "ejari.png", "ejari_certificate", extra)

    body = _upload(client, ["photo.png", "ejari.png"]).json()

    ejari = next(d for d in body["documents"] if d["document_type"] == "ejari_certificate")
    assert ejari["error"] == expected
    assert body["form_data"][0]["status"] == "success"
    assert body["status"] == ("success" if expected is None else "failed")


def test_a_company_document_must_be_for_the_licence_company(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    company_fields = COMPANY_DATA_FIELDS + ["License_Members", "Holder_Name"]
    _stub(
        monkeypatch,
        segments={
            "tl.png": [_region("trade_license", 1)], "eid.png": [_region("national_id", 1)],
            "ejari.png": [_region("ejari_certificate", 1)],
        },
        answers={
            "tl.png": _raw(["trade_license"], company_fields, Company_Name_En="Al Frejun L.L.C-FZ",
                           Trade_License_Number="2527004.01", License_Members="Bhargava Subhash Kalluri (Manager)"),
            "eid.png": _raw(["national_id"], company_fields, Signatory_Name_En="Bhargava Subhash Kalluri"),
            "ejari.png": _raw(["ejari_certificate"], company_fields, Company_Name_En="Beta Foods LLC"),
        },
    )

    body = _upload(client, ["tl.png", "eid.png", "ejari.png"], tenant_type="company").json()

    assert body["form_data"][0]["status"] == "success"
    ejari = next(d for d in body["documents"] if d["document_type"] == "ejari_certificate")
    assert ejari["error"] == (
        "'Ejari certificate' is not for the company on the trade licence or initial approval. "
        "Please upload the company's own Ejari certificate, or remove it."
    )


def test_a_passports_home_country_number_never_becomes_the_national_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An Egyptian passport prints a national number; the tenant's National ID is the Emirates ID.
    # Occupation is taken from the Emirates ID over the older passport.
    _stub(
        monkeypatch,
        segments={"pp.png": [_region("passport", 1)], "eid.png": [_region("national_id", 1)]},
        answers={
            "pp.png": _raw(["passport"], National_Id="27105041401018", Passport_Number="A28334482",
                           Occupation="ARCHITECTURAL ENGINEER", **MAGDY),
            "eid.png": _raw(["national_id"], National_Id="784-1971-6060254-2", Occupation="Chief Executive Officer", **MAGDY),
        },
    )
    sent: list[list[dict[str, Any]]] = []

    async def fake_cross(**kwargs: Any) -> list[str]:
        sent.append(kwargs["documents"])
        return []

    monkeypatch.setattr(settings, "validation_enabled", True)
    monkeypatch.setattr(get_gemini_service(), "verify_document", lambda **kwargs: _no_issues())
    monkeypatch.setattr(get_gemini_service(), "verify_documents_cross", fake_cross)

    body = _upload(client, ["pp.png", "eid.png"]).json()

    [tenant] = body["form_data"]
    assert (tenant["national_id"], tenant["occupation"]) == ("784-1971-6060254-2", "Chief Executive Officer")
    passport = next(document for document in sent[0] if document["document_type"] == "passport")
    assert passport["data"]["National_Id"] is None


async def _no_issues() -> list[str]:
    return []


def test_a_numbered_document_is_named_once_and_the_fix_uses_its_plain_type() -> None:
    from app.services.leasing.tenant_results import _unmatched_message
    from app.services.type_detection.tenants import Unit

    unit = Unit(key=(3,), document_type="ejari_certificate", data={"Tenant_Name_En": "Someone Else"})
    assert _unmatched_message(unit, "Ejari certificate 2", company=False, several=False) == (
        "'Ejari certificate 2' doesn't mention the tenant. Please upload the tenant's own Ejari certificate, or remove it."
    )



# --- company: a trade licence OR an initial approval; the approval has its own form_data keys -----

APPROVAL_FIELDS = COMPANY_DATA_FIELDS + [
    "Initial_Approval_Number", "Initial_Approval_Issue_Date", "Initial_Approval_Expiry", "Initial_Approval_Issuer",
    "License_Members", "Holder_Name",
]


def test_a_company_with_only_an_initial_approval_is_accepted(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(
        monkeypatch,
        segments={"approval.png": [_region("initial_approval", 1)], "eid.png": [_region("national_id", 1)]},
        answers={
            "approval.png": _raw(["initial_approval"], APPROVAL_FIELDS,
                                 Initial_Approval_Number="1788073", Initial_Approval_Issue_Date="21/07/2026",
                                 Initial_Approval_Expiry="21/01/2027", Initial_Approval_Issuer="Dep. of Economic Development",
                                 License_Members="Cyro Alessandro Martins Cardoso (Partner); Alaa Mohamed Eltom (Partner)"),
            "eid.png": _raw(["national_id"], APPROVAL_FIELDS, Signatory_Name_En="Cyro Alessandro Martins Cardoso",
                            Signatory_National_Id="784-1997-0472920-3"),
        },
    )

    body = _upload(client, ["approval.png", "eid.png"], tenant_type="company").json()

    assert body["status"] == "success"
    [tenant] = body["form_data"]
    assert (tenant["initial_approval_number"], tenant["initial_approval_issue_date"], tenant["initial_approval_expiry"],
            tenant["initial_approval_issuer"]) == ("1788073", "21/07/2026", "21/01/2027", "Dep. of Economic Development")
    # The trade licence keys stay empty - an approval is not a licence.
    assert tenant["trade_license_number"] is None and "trade_license_number" in body["missing"][0]["fields"]
    assert (tenant["signatory_name_en"], tenant["signatory_designation"]) == ("Cyro Alessandro Martins Cardoso", "Partner")


def test_a_licence_and_an_approval_of_one_company_are_one_tenant(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    # The approval prints no company name, but lists the licence's own manager.
    _stub(
        monkeypatch,
        segments={
            "tl.png": [_region("trade_license", 1)], "approval.png": [_region("initial_approval", 1)],
            "eid.png": [_region("national_id", 1)],
        },
        answers={
            "tl.png": _raw(["trade_license"], APPROVAL_FIELDS, Company_Name_En="Alpha Trading LLC",
                           Trade_License_Number="1001", License_Members="Cyro Alessandro Martins Cardoso (Manager)"),
            "approval.png": _raw(["initial_approval"], APPROVAL_FIELDS, Initial_Approval_Number="1788073",
                                 License_Members="Cyro Alessandro Martins Cardoso (Partner)"),
            "eid.png": _raw(["national_id"], APPROVAL_FIELDS, Signatory_Name_En="Cyro Alessandro Martins Cardoso"),
        },
    )

    body = _upload(client, ["tl.png", "approval.png", "eid.png"], tenant_type="company").json()

    assert body["status"] == "success"
    [tenant] = body["form_data"]
    assert (tenant["trade_license_number"], tenant["initial_approval_number"]) == ("1001", "1788073")
    assert all(document["tenant_ids"] == ["tenant_1"] for document in body["documents"])


def test_individual_responses_have_no_approval_keys(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, segments={"eid.png": [_region("national_id", 1)]}, answers={"eid.png": _raw(["national_id"])})

    body = _upload(client, ["eid.png"]).json()

    assert not any(key.startswith("initial_approval") for key in body["form_data"][0])
