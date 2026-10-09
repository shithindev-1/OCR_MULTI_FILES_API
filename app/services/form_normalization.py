"""Post-processing of the merged form_data values, for both tenant types, after every check has
passed - so cross-verification still compares what was actually printed on each document.

  - Gender: M / F -> Male / Female (any case); Male / Female in any case -> Male / Female.
  - Dates: whatever format a document printed, reported as DD/MM/YYYY.
  - Countries and nationalities: a code, another spelling or a nationality ("SY", "SYR", "syrian")
    -> the exact name in the client's country list ("Syria").

A value that cannot be recognised is left exactly as it was read, never blanked. An unrecognised
nationality or country is also saved as a new name and matched in the background, so it is
converted from the next request on (app.services.country_learning).
"""

import re
from datetime import date
from typing import Any

from app.services.countries import country_name
from app.services.country_learning import learn_in_background, learned_country_name

_GENDERS = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}

# Fields holding a country, in either tenant type's field list.
COUNTRY_FIELDS = frozenset({
    "Nationality",
    "Home_Country",
    "Resident_Country",
    "Passport_Issuing_Location",
    "Signatory_Passport_Issuing_Location",
})
# The ones whose unrecognised values are saved as new names. Issuing location is left out: it often
# holds an authority or a city ("IMMIGRATION AND CITIZENSHIP SERVICE", "Dubai"), not a country.
LEARNED_COUNTRY_FIELDS = frozenset({"Nationality", "Home_Country", "Resident_Country"})

# Matched on a word's first three letters: "Apr", "APRIL", "Sept" and "JANV" (French) all count.
_MONTHS = {name: number for number, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1,
)}
_TIME = re.compile(r"\d{1,2}:\d{2}(:\d{2})?")
_TOKEN = re.compile(r"\d+|[^\W\d_]+")


def is_date_field(field: str) -> bool:
    """Date_Of_Birth, Visa_Start_Date, Trade_License_Issue_Date, Passport_Expiry, ..."""
    return "Date" in field or field.endswith("_Expiry")


def _full_year(year: int, *, birth: bool) -> int:
    """Two-digit years are taken as 20YY - or 19YY for a birth date that would otherwise be in the future."""
    if year >= 100:
        return year
    year += 2000
    return year - 100 if birth and year > date.today().year else year


def _build(day: int, month: int, year: int, *, birth: bool) -> date | None:
    try:
        return date(_full_year(year, birth=birth), month, day)
    except ValueError:
        return None


def _parse_date(value: str, *, birth: bool) -> date | None:
    tokens = _TOKEN.findall(_TIME.sub(" ", value))
    numbers = [token for token in tokens if token.isdigit()]
    month = next((_MONTHS[token[:3].lower()] for token in tokens if token[:3].lower() in _MONTHS), None)

    if month is not None:
        # "03 APR 2020", "April 3, 2020", "2020 Apr 03": the 4-digit number is the year, the other the day.
        if len(numbers) < 2:
            return None
        first, second = numbers[0], numbers[1]
        if len(first) == 4 or (len(second) <= 2 and int(first) > 31):
            return _build(int(second), month, int(first), birth=birth)
        return _build(int(first), month, int(second), birth=birth)

    if len(numbers) == 1 and len(numbers[0]) == 8:
        # "20200403" or "03042020".
        digits = numbers[0]
        if digits[:2] in ("19", "20"):
            found = _build(int(digits[6:]), int(digits[4:6]), int(digits[:4]), birth=birth)
            if found:
                return found
        return _build(int(digits[:2]), int(digits[2:4]), int(digits[4:]), birth=birth)

    if len(numbers) < 3:
        return None
    first, second, third = numbers[:3]
    if len(first) == 4:
        # "2020-04-03": year first.
        return _build(int(third), int(second), int(first), birth=birth)
    # Day first, as UAE documents print it; month first only when day-first is impossible ("04/23/2020").
    return (_build(int(first), int(second), int(third), birth=birth)
            or _build(int(second), int(first), int(third), birth=birth))


def format_date(value: str, *, birth: bool = False) -> str:
    """Any printed date as DD/MM/YYYY; unchanged if it cannot be read as a date."""
    parsed = _parse_date(value, birth=birth)
    return parsed.strftime("%d/%m/%Y") if parsed else value


def format_gender(value: str) -> str:
    return _GENDERS.get(value.strip().lower(), value)


def normalize_form_data(data: dict[str, Any]) -> dict[str, Any]:
    """Formats every gender, date and country value in merged form_data in place, and returns it."""
    unrecognised: list[str] = []
    for field, value in data.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if field == "Gender":
            data[field] = format_gender(value)
        elif is_date_field(field):
            data[field] = format_date(value, birth=field == "Date_Of_Birth")
        elif field in COUNTRY_FIELDS:
            name = country_name(value) or learned_country_name(value)
            if name:
                data[field] = name
            elif field in LEARNED_COUNTRY_FIELDS:
                unrecognised.append(value)
    learn_in_background(unrecognised)
    return data
