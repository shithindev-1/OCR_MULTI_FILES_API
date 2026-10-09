"""Country lookup for form_data post-processing, over the curated lists in app/data.

Every country is reported by its exact name, the names in country_codes.json, whatever a document
printed: "SY", "SYR", "Syrian Arab Republic" and "syrian" all become "Syria". Matching ignores
case, dots and extra spaces.

Reference data (edit these files, not this module):
  - country_codes.json  code -> exact name: the client's 2-letter codes, 3-letter passport codes
                        and short forms such as UAE, UK, KSA
  - nationalities.json  nationality or other country name -> exact name

Both files are loaded and checked once, when the app starts: an entry pointing at a name that is not
one of the exact names stops startup instead of silently returning a name outside the client's list.
Values neither file knows are learned at runtime - see app.services.country_learning.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_CODE = re.compile(r"^[a-z]{2,3}$")


def lookup_key(value: str) -> str:
    """Case, dots and spacing ignored: "U.A.E." and "uae", "SYRIAN" and "Syrian" look the same."""
    return re.sub(r"\s+", " ", value.replace(".", "")).strip(" ,;:-").lower()


def is_code(value: str) -> bool:
    """A 2- or 3-letter value ("SY", "SYR", "U.A.E") is a code; anything longer is a name."""
    return bool(_CODE.fullmatch(lookup_key(value)))


def _load(filename: str):
    with (DATA_DIR / filename).open(encoding="utf-8") as handle:
        return json.load(handle)


def _build() -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    codes = _load("country_codes.json")
    exact_names = tuple(dict.fromkeys(entry["name"] for entry in codes))
    # A code listed twice keeps its first country (PA: Panama, not Palestine, which is still
    # matched by name).
    by_code: dict[str, str] = {}
    for entry in codes:
        by_code.setdefault(entry["code"].upper(), entry["name"])

    lookup: dict[str, str] = {}
    for alias, name in _load("nationalities.json").items():
        if name not in exact_names:
            raise ValueError(f"nationalities.json: '{alias}' points to '{name}', which is not in country_codes.json")
        lookup[lookup_key(alias)] = name
    for name in exact_names:
        lookup[lookup_key(name)] = name
    # Codes last, so they win over a same-spelled name.
    for code, name in by_code.items():
        lookup[lookup_key(code)] = name
    return exact_names, lookup, by_code


EXACT_NAMES, _LOOKUP, _BY_CODE = _build()


def name_for_code(code: str) -> str | None:
    """The exact country name for a code in country_codes.json ("SY" -> "Syria"), or None."""
    return _BY_CODE.get(code.strip().upper())


def country_name(value: str) -> str | None:
    """The exact country name for a code, name or nationality; None if the curated lists do not know
    it. A bilingual value such as "Indian / هندي" is matched on whichever part is known."""
    found = _LOOKUP.get(lookup_key(value))
    if found:
        return found
    for part in re.split(r"[/|]", value):
        found = _LOOKUP.get(lookup_key(part))
        if found:
            return found
    return None
