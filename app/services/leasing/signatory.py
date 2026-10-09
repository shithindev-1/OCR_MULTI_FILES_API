"""A company signatory's designation, taken from their own role on the trade licence."""

import re
from typing import Any

from app.services.names import same_name

_MEMBER = re.compile(r"^\s*(?P<name>[^()]+?)\s*\((?P<role>[^()]*)\)\s*$")


def apply_signatory_role_from_license(merged_data: dict[str, Any], documents: list[dict[str, Any]]) -> None:
    """Sets the signatory's designation to their own role on the trade licence, when the licence lists them.

    Each document is read on its own, so a licence's reading cannot know which of its people is the
    signatory; this matches the signatory's name (from their national ID, or the holder name on their
    passport/visa) against the licence's "Name (Role)" list. The licence role is the person's
    official role in this company, so it wins over a job title printed on their own ID or visa;
    without a match, the designation read from their own documents stays as it is.
    """
    signatory = merged_data.get("Signatory_Name_En") or next(
        (document["data"].get("Holder_Name") for document in documents if document["data"].get("Holder_Name")), None,
    )
    if not signatory:
        return
    for document in documents:
        for member in str(document["data"].get("License_Members") or "").split(";"):
            match = _MEMBER.match(member)
            if not match or not same_name(signatory, match["name"]):
                continue
            # "Manager / مدير" -> "Manager": the English part, as the other fields are English.
            role = next((part.strip() for part in match["role"].split("/") if part.strip()), "")
            if role:
                merged_data["Signatory_Designation"] = role
                return
