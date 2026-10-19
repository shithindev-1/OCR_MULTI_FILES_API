"""One cross-check message per document, written from that document's point of view.

The type-detection cross-check returns what disagrees - the documents and the detail - and this
module words it for each document, so a passport and a visa that disagree each get their own
message instead of one shared sentence.
"""

from typing import Any

from app.services.post_processing import document_label

_DETAIL_WORDS = {
    "name": "name",
    "date_of_birth": "date of birth",
    "nationality": "nationality",
    "gender": "gender",
    "passport_number": "passport number",
    "national_id_number": "ID number",
    "visa_number": "visa number",
    "company_name": "company name",
    "licence_number": "licence number",
}
# The document type a number belongs to: on that document the number is its own; on any other
# document it is a copy (a visa printing the passport number).
_NUMBER_OWNER = {
    "passport_number": {"passport"},
    "national_id_number": {"national_id"},
    "visa_number": {"visa"},
    "licence_number": {"trade_license", "initial_approval"},
}
_COMPANY_DETAILS = {"company_name", "licence_number"}
_LICENCE_TYPES = {"trade_license", "initial_approval"}


def _quoted(labels: list[str]) -> str:
    quoted = [f"'{label}'" for label in labels]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]


def resolve_documents(names: list[str], types: dict[str, str]) -> list[str]:
    """The labels a conflict names. The model is asked for the labels ("Visa"), but may answer with
    another case or the document type ("visa"); a type counts only when one document has it."""
    by_lower = {label.lower(): label for label in types}
    resolved: list[str] = []
    for name in names:
        key = str(name).strip().strip("'").lower()
        label = by_lower.get(key)
        if label is None:
            of_type = [lbl for lbl, kind in types.items() if kind == key.replace(" ", "_")]
            label = of_type[0] if len(of_type) == 1 else None
        if label and label not in resolved:
            resolved.append(label)
    return resolved


def message_for(document: str, conflict: dict[str, Any], types: dict[str, str]) -> str:
    """The message for `document` (its label) about one conflict. `types` maps label -> document type."""
    others = [label for label in conflict["documents"] if label != document]
    detail = conflict.get("detail") or "other"
    kind = document_label(types.get(document, ""))
    if detail == "other" or not others:
        return conflict.get("message") or (
            f"'{document}' doesn't match {_quoted(others) or 'the other documents'}. Please upload documents that match."
        )
    if detail == "listed_person":
        if types.get(document) in _LICENCE_TYPES:
            return (f"'{document}' doesn't list the person on {_quoted(others)}. "
                    "Please upload the documents of a person it lists.")
        return (f"The person on '{document}' is not listed on {_quoted(others)}. "
                "Please upload the documents of a person it lists.")
    words = _DETAIL_WORDS.get(detail, detail.replace("_", " "))
    owners = _NUMBER_OWNER.get(detail)
    if owners:
        if types.get(document) in owners:
            return (f"The {words} on '{document}' is different from the one on {_quoted(others)}. "
                    f"Please check you uploaded the right {kind}.")
        owner = next((label for label in others if types.get(label) in owners), None)
        if owner:
            return (f"'{document}' shows a different {words} than '{owner}'. "
                    f"Please upload the {kind} issued for this {document_label(types[owner])}.")
    same = "company" if detail in _COMPANY_DETAILS else "person"
    return (f"'{document}' shows a different {words} from {_quoted(others)}. "
            f"Please upload documents of the same {same}.")
