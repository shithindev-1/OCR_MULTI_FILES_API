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
    assert sorted(extracted) == ["scan_1_national_id.jpg", "scan_2_passport.jpg"]
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
    assert response.json()["detail"].startswith("A trade licence is required for a company.")


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
        "None of the people listed on this trade licence were uploaded. "
        "Please upload the ID documents of a person listed on the licence."
    )
    assert person["error"].startswith("The person on 'National ID' is not listed on any uploaded trade licence.")


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
    assert (passport.document_type, passport.pages, passport.extension) == ("passport", [1], ".jpg")


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
    assert zaid["error"].startswith("The person on 'National ID 3' is not listed on any uploaded trade licence.")


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
