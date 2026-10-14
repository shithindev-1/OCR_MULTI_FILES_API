"""form_data post-processing: gender, dates, country names, and learning new country values."""

import asyncio
import json

import pytest

from app.core.config import settings
from app.services import country_learning
from app.services.countries import EXACT_NAMES, country_name
from app.services.form_normalization import (
    format_date,
    format_gender,
    normalize_form_data,
)
from app.services.gemini import get_gemini_service


@pytest.mark.parametrize(("value", "expected"), [
    ("m", "Male"), ("M", "Male"), ("MALE", "Male"), ("male", "Male"),
    ("f", "Female"), ("F", "Female"), ("fEmAlE", "Female"),
    ("X", "X"),
])
def test_gender(value, expected):
    assert format_gender(value) == expected


@pytest.mark.parametrize(("value", "expected"), [
    ("04.03.2001", "04-03-2001"),
    ("2030-04-15", "15-04-2030"),
    ("15 APR 2030", "15-04-2030"),
    ("April 15, 2030", "15-04-2030"),
    ("15 JAN/JANV 2030", "15-01-2030"),
    ("15-04-30", "15-04-2030"),
    ("20300415", "15-04-2030"),
    ("2030-04-15T00:00:00", "15-04-2030"),
    ("04/23/2030", "23-04-2030"),  # day-first impossible, so month-first
    ("not a date", "not a date"),
])
def test_dates_become_dd_mm_yyyy_with_dashes(value, expected):
    assert format_date(value) == expected


def test_two_digit_birth_year_is_never_in_the_future():
    assert format_date("01/02/95", birth=True) == "01-02-1995"


@pytest.mark.parametrize(("value", "expected"), [
    ("SY", "Syria"), ("syr", "Syria"), ("Syrian", "Syria"), ("SYRIAN ARAB REPUBLIC", "Syria"),
    ("ae", "United Arab Emirates"), ("U.A.E.", "United Arab Emirates"), ("Emirati", "United Arab Emirates"),
    ("ETH", "Ethiopia"), ("ETHIOPIAN", "Ethiopia"),
    ("PA", "Panama"), ("Palestinian", "Palestine"),
    ("Indian / هندي", "India"),
    ("Dubai", None),
])
def test_country_codes_and_nationalities_become_exact_names(value, expected):
    assert country_name(value) == expected


def test_every_curated_name_is_an_exact_name():
    assert all(country_name(name) in EXACT_NAMES for name in EXACT_NAMES)


def test_normalize_form_data_formats_each_kind_of_field():
    data = normalize_form_data({
        "Gender": "F", "Date_Of_Birth": "04.03.2001", "Passport_Expiry": "26.01.2031",
        "Nationality": "ETHIOPIAN", "Passport_Issuing_Location": "IMMIGRATION AND CITIZENSHIP SERVICE",
        "Tenant_Name_En": "REDIET DAGIM TILAHUN", "Visa_Number": None,
    })
    assert data == {
        "Gender": "Female", "Date_Of_Birth": "04-03-2001", "Passport_Expiry": "26-01-2031",
        "Nationality": "Ethiopia", "Passport_Issuing_Location": "IMMIGRATION AND CITIZENSHIP SERVICE",
        "Tenant_Name_En": "REDIET DAGIM TILAHUN", "Visa_Number": None,
    }


@pytest.fixture
def learning(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "country_learning_enabled", True)
    monkeypatch.setattr(settings, "country_learning_dir", str(tmp_path))
    asked: list[str] = []
    # Kosovar: a real country (XK) that is not in the client's list.
    answers = {"Syrian Arab Rep": "SY", "SYA": "SY", "Atlantis": None, "Kosovar": "XK"}

    async def fake_country_code(*, value, kind):
        asked.append(value)
        return answers[value]

    monkeypatch.setattr(get_gemini_service(), "country_code", fake_country_code)
    return tmp_path, asked


async def _normalize_and_wait(data):
    result = normalize_form_data(data)
    await asyncio.gather(*country_learning._tasks)
    return result


def test_new_values_are_returned_as_read_then_saved_to_their_own_list(learning):
    folder, asked = learning
    first = asyncio.run(_normalize_and_wait(
        {"Nationality": "Syrian Arab Rep", "Home_Country": "SYA", "Resident_Country": "Atlantis"},
    ))
    assert first == {"Nationality": "Syrian Arab Rep", "Home_Country": "SYA", "Resident_Country": "Atlantis"}

    codes = json.loads((folder / country_learning.CODES_FILE).read_text(encoding="utf-8"))
    names = json.loads((folder / country_learning.NATIONALITIES_FILE).read_text(encoding="utf-8"))
    assert codes["SYA"]["name"] == "Syria"
    assert names["Syrian Arab Rep"]["name"] == "Syria"
    assert names["Syrian Arab Rep"]["code"] == "SY"
    assert names["Atlantis"]["name"] is None

    second = asyncio.run(_normalize_and_wait(
        {"Nationality": "syrian arab rep", "Home_Country": "sya", "Resident_Country": "Atlantis"},
    ))
    assert second == {"Nationality": "Syria", "Home_Country": "Syria", "Resident_Country": "Atlantis"}
    assert sorted(asked) == ["Atlantis", "SYA", "Syrian Arab Rep"]  # each value asked about once


def test_a_country_outside_the_client_list_is_saved_without_a_name(learning):
    folder, _ = learning
    assert asyncio.run(_normalize_and_wait({"Nationality": "Kosovar"})) == {"Nationality": "Kosovar"}
    entry = json.loads((folder / country_learning.NATIONALITIES_FILE).read_text(encoding="utf-8"))["Kosovar"]
    assert entry["name"] is None and entry["code"] == "XK"


def test_issuing_location_is_never_saved_as_a_new_name(learning):
    folder, asked = learning
    asyncio.run(_normalize_and_wait({"Passport_Issuing_Location": "IMMIGRATION AND CITIZENSHIP SERVICE"}))
    assert asked == []
    assert not any(folder.iterdir())


def test_a_hand_edited_entry_is_used_without_a_restart(learning):
    folder, _ = learning
    (folder / country_learning.NATIONALITIES_FILE).write_text(
        json.dumps({"Atlantis": {"name": "Greece", "source": "manual", "added": "2026-10-08"}}), encoding="utf-8",
    )
    assert asyncio.run(_normalize_and_wait({"Nationality": "Atlantis"})) == {"Nationality": "Greece"}
