"""The document-type registry and the focused (per-type) extraction it drives."""

from typing import Any

import pytest

from app.schemas.ocr import (
    COMPANY_EXTRACTION_FIELDS,
    DETECT_DOCUMENT_TYPES,
    INDIVIDUAL_EXTRACTION_FIELDS,
    PROPERTY_FIELDS,
)
from app.services import document_registry as registry
from app.services.gemini import prompts


def test_every_supported_type_has_an_entry() -> None:
    assert set(registry.REGISTRY) == set(DETECT_DOCUMENT_TYPES)


@pytest.mark.parametrize("document_type", sorted(DETECT_DOCUMENT_TYPES))
@pytest.mark.parametrize("company", [False, True])
def test_every_field_is_known_and_defined(document_type: str, company: bool) -> None:
    fields = registry.extraction_fields(document_type, company)
    # Property details are read on the type-detection endpoints only, to match a rental index.
    known = (COMPANY_EXTRACTION_FIELDS if company else INDIVIDUAL_EXTRACTION_FIELDS) + PROPERTY_FIELDS
    assert fields and set(fields) <= set(known)
    rules = registry.field_rules(document_type, company)
    definitions = registry.COMPANY_FIELD_RULES if company else registry.INDIVIDUAL_FIELD_RULES
    for field in fields:
        assert field in definitions or field in registry._ADDRESS_FIELDS, field
        assert field in rules or field in registry._ADDRESS_FIELDS


def test_a_focused_instruction_asks_only_for_its_own_fields() -> None:
    instruction, fields = prompts.focused_system_instruction("passport", company=False)
    assert "Passport_Number: <value or null>" in instruction
    assert "Visa_Number: <value or null>" not in instruction and "National_Id: <value or null>" not in instruction
    assert "Document_Types_Present" in instruction and "may be wrong" in instruction
    assert prompts.focused_system_instruction("other", company=False) is None


def test_licence_entries_read_members_for_an_individual_and_company_details_for_a_company() -> None:
    assert registry.extraction_fields("trade_license", company=False) == ["License_Members"]
    company = registry.extraction_fields("initial_approval", company=True)
    assert "Initial_Approval_Number" in company and "Trade_License_Number" not in company


def test_required_details_and_link_numbers_come_from_the_registry() -> None:
    assert "two lines of letters and numbers" in registry.required_details("passport")
    assert "light-grey" in prompts.covered_parts_rules("passport") and "passport number" in prompts.covered_parts_rules("passport")
    assert registry.link_numbers("visa", company=False) == ("Visa_Number", "Passport_Number", "National_Id")
    assert registry.link_numbers("trade_license", company=True) == ("Trade_License_Number",)


def test_detect_endpoints_read_focused_and_old_endpoints_do_not(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from app.core.config import settings
    from app.main import app
    from app.schemas.ocr import DATA_FIELDS
    from app.services import blob_storage
    from app.services.gemini import get_gemini_service

    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: None)
    calls: list[dict[str, Any]] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        # The ID number is read, so the focused reading needs no second, general reading.
        return {"data": {**dict.fromkeys(DATA_FIELDS), "National_Id": "784-1990-1234567-1"}, "documentComplete": True,
                "documentTypesPresent": ["national_id"],
                "frontVisible": True, "backVisible": True}

    async def fake_segment(pages: list[bytes]) -> list[dict[str, Any]]:
        return [{"regions": [{"document_type": "national_id", "side": "single", "document_number": 1,
                              "box_2d": [0, 0, 1000, 1000], "covered_by": []}]}]

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(get_gemini_service(), "segment_documents", fake_segment)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), "white").save(buffer, format="PNG")
    png = buffer.getvalue()
    client = TestClient(app)

    client.post("/api/v1/ocr/leasing/detect/upload", files=[("file", ("eid.png", png, "image/png"))],
                data={"tenant_type": "individual", "document_refnumber": "r1"})
    client.post("/api/v1/ocr/leasing/upload", files=[("file", ("eid.png", png, "image/png"))],
                data={"tenant_type": "individual", "document_refnumber": "r2", "document_id": "1",
                      "document_type": "national_id", "document_name": "nid"})

    assert [call.get("focused", False) for call in calls] == [True, False]


def test_a_place_field_with_no_letters_is_dropped_from_a_focused_reading() -> None:
    from app.services.ocr_pipeline import drop_numeric_places

    data = {"Passport_Issuing_Location": "3", "Nationality": "EGYPTIAN", "Home_City": "12-34",
            "Passport_Number": "A28334482", "Resident_City": "Dubai"}
    assert drop_numeric_places(data) == {"Passport_Issuing_Location": None, "Nationality": "EGYPTIAN",
                                         "Home_City": None, "Passport_Number": "A28334482", "Resident_City": "Dubai"}


def test_issuing_location_is_defined_as_a_place_never_an_office_code() -> None:
    rules = registry.field_rules("passport", company=False)
    assert "never a place" in rules and "Issuing Office" in rules
    assert "never the value of a name, place, nationality or date field" in rules


def test_decoding_settings_are_sent_on_every_call() -> None:
    from app.services.gemini import get_gemini_service

    config = get_gemini_service()._generation_config()
    assert (config.temperature, config.seed, config.top_p, config.top_k) == (0.0, 0, 1.0, 1)



# --- safety net: a focused reading that misses the document's own number is read again --------------

def _reading(**values: Any) -> dict[str, Any]:
    from app.schemas.ocr import DATA_FIELDS

    return {"data": {**dict.fromkeys(DATA_FIELDS), **values}, "documentComplete": True,
            "documentTypesPresent": ["passport"], "frontVisible": True, "backVisible": True}


def _read(monkeypatch: pytest.MonkeyPatch, answers: dict[bool, Any], document_type: str = "passport") -> tuple[dict, list]:
    import asyncio

    from app.services import result_cache
    from app.services.gemini import get_gemini_service
    from app.services.ocr_pipeline import extract_and_type_check

    calls: list[bool] = []

    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        focused = kwargs.get("focused", False)
        calls.append(focused)
        answer = answers[focused]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(result_cache, "get", lambda key: None)
    monkeypatch.setattr(result_cache, "put", lambda key, value: None)
    document = asyncio.run(extract_and_type_check(
        request_id="r", position="1/1", content=b"x", filename="p.png", mime_type="image/png", sha256="s",
        document_name="Passport", document_type=document_type, started_at=0.0, check_type=False, focused=True,
    ))
    return document["data"], calls


def test_a_focused_reading_missing_its_own_number_is_completed_by_a_general_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    focused = _reading(Tenant_Name_En="Magdy Ali")
    general = _reading(Tenant_Name_En="MAGDY ALI", Passport_Number="A28334482", Passport_Issuing_Location="3",
                       Visa_Number="201/2023/1")
    data, calls = _read(monkeypatch, {True: focused, False: general})

    assert calls == [True, False]
    assert data["Passport_Number"] == "A28334482"      # filled from the general reading
    assert data["Tenant_Name_En"] == "Magdy Ali"       # a focused value is never replaced
    assert data["Passport_Issuing_Location"] is None   # an office number is still never a place
    assert data.get("Visa_Number") is None             # only the passport's own fields are filled


def test_a_focused_reading_with_its_own_number_is_not_read_again(monkeypatch: pytest.MonkeyPatch) -> None:
    data, calls = _read(monkeypatch, {True: _reading(Passport_Number="A28334482"), False: _reading()})

    assert calls == [True] and data["Passport_Number"] == "A28334482"


def test_a_failed_general_reading_keeps_the_focused_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services.gemini.client import GeminiServiceError

    data, calls = _read(monkeypatch, {True: _reading(Tenant_Name_En="Magdy Ali"), False: GeminiServiceError("down")})

    assert calls == [True, False] and data["Tenant_Name_En"] == "Magdy Ali" and data["Passport_Number"] is None


def test_the_back_of_an_id_shown_alone_need_not_print_its_number(monkeypatch: pytest.MonkeyPatch) -> None:
    back = {**_reading(Occupation="Engineer"), "frontVisible": False, "documentTypesPresent": ["national_id"]}
    data, calls = _read(monkeypatch, {True: back, False: _reading()}, document_type="national_id")

    assert calls == [True] and data["Occupation"] == "Engineer"



def test_detect_readings_may_report_a_rental_index_and_old_ones_are_unchanged() -> None:
    # The shared type list says "exactly these spellings and no others": without rental_index in it, a
    # rental index was reported as "other" and rejected by the type check.
    for document_type in ("rental_index", "passport", "ejari_certificate"):
        instruction, _ = prompts.focused_system_instruction(document_type, company=False)
        assert "rental_index for a rental index result" in instruction
    assert "rental_index" not in prompts.SYSTEM_INSTRUCTION
    assert "rental_index" not in prompts.COMPANY_SYSTEM_INSTRUCTION



def test_a_rental_index_is_never_reported_as_expired_by_its_contract_end_date() -> None:
    assert "never report the rental index, or that date, as expired" in prompts.covered_parts_rules("rental_index")
    assert "RENTAL INDEX" not in prompts.covered_parts_rules("passport")



def test_page_numbering_is_never_a_problem_on_the_detect_self_check() -> None:
    # A 7-page bank statement printing "Page 1 of 1" on its pages was rejected for its page count.
    assert "never report anything about page numbers, page counts" in prompts.SHORT_SELF_MESSAGES
    assert "PAGES - never report" not in prompts.SELF_VERIFICATION_PROMPT
