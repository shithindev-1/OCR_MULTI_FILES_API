import pytest

from app.services.post_processing import (
    document_type_matches,
    normalize_response,
    parse_extraction_text,
)


def test_parse_extraction_text_maps_plain_lines_to_fields() -> None:
    parsed = parse_extraction_text(
        "\n".join(
            [
                "Tenant_Name_En: Jane Doe",
                "Tenant_Name_Ar: جين دو",
                "Date_Of_Birth: 1990-01-15",
                "Nationality: null",
                "National_Id: 784-1990-1234567-1",
                "Front_Side_Visible: true",
                "Back_Side_Visible: true",
            ]
        )
    )

    assert parsed["data"]["Tenant_Name_En"] == "Jane Doe"
    assert parsed["data"]["Date_Of_Birth"] == "1990-01-15"
    assert parsed["data"]["National_Id"] == "784-1990-1234567-1"
    assert parsed["data"]["Nationality"] is None
    # Fields the model never mentioned still exist, as null.
    assert parsed["data"]["Passport_Number"] is None
    assert parsed["documentComplete"] is True
    assert parsed["frontVisible"] is True
    assert parsed["backVisible"] is True


def test_parse_extraction_text_reports_which_side_was_seen() -> None:
    parsed = parse_extraction_text(
        "\n".join(["Tenant_Name_En: Jane Doe", "Front_Side_Visible: true", "Back_Side_Visible: false"])
    )

    assert parsed["documentComplete"] is False
    assert parsed["frontVisible"] is True
    assert parsed["backVisible"] is False


@pytest.mark.parametrize("placeholder", ["null", "N/A", "not visible", "-", "Unknown", "", "none"])
def test_parse_extraction_text_treats_placeholders_as_null(placeholder: str) -> None:
    parsed = parse_extraction_text(f"Tenant_Name_En: {placeholder}\nNationality: UAE")

    assert parsed["data"]["Tenant_Name_En"] is None
    assert parsed["data"]["Nationality"] == "UAE"


def test_parse_extraction_text_survives_markdown_and_prose_contamination() -> None:
    parsed = parse_extraction_text(
        "\n".join(
            [
                "Here are the extracted fields:",
                "```",
                "## Extraction",
                "- **Tenant_Name_En**: Jane Doe",
                "* `Nationality`: UAE",
                "Note: some values were hard to read.",
                "Back_Side_Visible: false",
                "```",
            ]
        )
    )

    assert parsed["data"]["Tenant_Name_En"] == "Jane Doe"
    assert parsed["data"]["Nationality"] == "UAE"
    assert parsed["documentComplete"] is False


def test_parse_extraction_text_keeps_colons_inside_values() -> None:
    parsed = parse_extraction_text("Employer_Name: Acme: Holdings LLC")

    assert parsed["data"]["Employer_Name"] == "Acme: Holdings LLC"


def test_parse_extraction_text_defaults_sides_to_present_when_absent() -> None:
    parsed = parse_extraction_text("Tenant_Name_En: Jane Doe")

    assert parsed["documentComplete"] is True
    assert parsed["frontVisible"] is True
    assert parsed["backVisible"] is True


def test_parse_extraction_text_rejects_a_response_with_no_recognisable_fields() -> None:
    with pytest.raises(ValueError, match="no recognisable field lines"):
        parse_extraction_text("I'm sorry, I can't help with that request.")


def test_normalize_response_strips_extra_keys_and_derives_missing_info() -> None:
    response = normalize_response(
        {
            "success": True,
            "data": {"Tenant_Name_En": "Jane Doe", "unexpected": "remove me"},
            "not_allowed": "remove me",
        },
        document_name="National ID",
        document_filename="id.pdf",
        document_type="national_id",
    )

    assert set(response) == {"success", "userName", "tenantType", "subscriptionId", "ocrReferenceId", "data", "missingInfo", "errorInfo"}
    assert response["data"]["Tenant_Name_En"] == "Jane Doe"
    assert "tenant_name_en" not in response["missingInfo"]
    assert "passport_number" in response["missingInfo"]
    assert "documentComplete" not in response


def test_document_complete_false_is_ignored_for_types_without_a_front_back_pair() -> None:
    response = normalize_response(
        {"documentComplete": False, "data": {"Tenant_Name_En": "Jane Doe"}},
        document_name="Ejari Certificate",
        document_filename="ejari.pdf",
        document_type="ejari_certificate",
    )

    assert response["errorInfo"] == []


def test_incomplete_document_gets_error_info() -> None:
    response = normalize_response(
        {"documentComplete": False, "data": {}},
        document_name="Emirates ID",
        document_filename="eid_front.jpg",
        document_type="national_id",
    )

    assert len(response["errorInfo"]) == 1
    assert response["errorInfo"][0]["DocumentName"] == "Emirates ID"


def test_incomplete_document_drops_whatever_was_extracted() -> None:
    # A one-sided National ID still yields real field values from the side that was visible - the
    # front's National_Id, say. Reporting "incomplete" alongside those values would let a caller
    # quietly use data from a document the API itself flagged as not good enough.
    response = normalize_response(
        {"documentComplete": False, "data": {"Tenant_Name_En": "Jane Doe", "National_Id": "784-1985-1234567-1"}},
        document_name="Emirates ID",
        document_filename="eid_front.jpg",
        document_type="national_id",
    )

    assert response["success"] is False
    assert all(value is None for value in response["data"].values())
    assert len(response["errorInfo"]) == 1


# --------------------------------------------------------------- document type validation


def _extraction(*extra_lines: str) -> str:
    return "\n".join(
        ["Tenant_Name_En: Jane Doe", "Passport_Number: C2323323",
         "Front_Side_Visible: true", "Back_Side_Visible: true", *extra_lines]
    )


def test_parse_reads_the_document_types_present_line() -> None:
    parsed = parse_extraction_text(_extraction("Document_Types_Present: passport, visa"))

    assert parsed["documentTypesPresent"] == ["passport", "visa"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Document_Types_Present: passport", ["passport"]),
        ("Document_Types_Present: Passport, National ID", ["passport", "national_id"]),
        ("Document_Types_Present: national-id; visa", ["national_id", "visa"]),
        # Common spellings of the same document must not read as different types.
        ("Document_Types_Present: Emirates ID", ["national_id"]),
        ("Document_Types_Present: residence visa", ["visa"]),
        ("Document_Types_Present: initial approval, salary certificate, tenancy contract",
         ["initial_approval", "salary_certificate", "tenancy_contract"]),
        ("Document_Types_Present: salary cert, tenancy agreement", ["salary_certificate", "tenancy_contract"]),
        # Anything outside the three identity documents collapses to "other".
        ("Document_Types_Present: other", ["other"]),
        ("Document_Types_Present: passport, tenancy contract", ["passport", "tenancy_contract"]),
        ("Document_Types_Present: none", []),
        # A duplicate is not two documents.
        ("Document_Types_Present: passport, Passport", ["passport"]),
    ],
)
def test_parse_normalises_type_names(line, expected) -> None:
    assert parse_extraction_text(_extraction(line))["documentTypesPresent"] == expected


def test_missing_types_line_is_unknown_not_empty() -> None:
    # Older responses, or a model that omits the line, must not read as "no types found".
    assert parse_extraction_text(_extraction())["documentTypesPresent"] == []


@pytest.mark.parametrize(
    ("requested", "present", "matches"),
    [
        ("passport", ["passport"], True),
        ("passport", ["national_id"], False),
        ("visa", ["passport", "national_id", "visa"], True),   # present among several
        ("visa", ["passport", "national_id"], False),          # absent among several
        # Every type the dropdown offers is checked, not just the identity documents: a file the
        # model classified as something else does not satisfy the selection.
        ("ejari_certificate", ["other"], False),
        ("ejari_certificate", ["ejari_certificate"], True),
        ("bank_statement", ["passport"], False),
        ("trade_license", ["trade_license", "other"], True),
        # "Other" as a selection means the caller is not asserting a type, so nothing can contradict it.
        ("other", [], True),
        ("other", ["passport"], True),
        ("initial_approval", ["other"], False),
        ("salary_certificate", ["salary_certificate"], True),
        ("tenancy_contract", ["passport"], False),
        # An unreported or malformed observation is unknown, never a rejection.
        ("passport", [], True),
        ("passport", None, True),
        ("passport", "passport", True),
    ],
)
def test_document_type_matches(requested, present, matches) -> None:
    assert document_type_matches(requested, present) is matches


def test_mismatch_returns_an_error_and_drops_the_extracted_data() -> None:
    raw = parse_extraction_text(_extraction("Document_Types_Present: passport"))

    response = normalize_response(
        raw, document_name="Tenant ID", document_filename="passport_scan.jpg",
        document_type="national_id", default_reference_id="req-1",
    )

    assert response["success"] is False
    assert all(value is None for value in response["data"].values())
    assert len(response["errorInfo"]) == 1
    error = response["errorInfo"][0]
    assert error["DocumentFileName"] == "passport_scan.jpg"
    assert "passport_scan.jpg" in error["DocumentError"]
    assert "National ID" in error["DocumentError"]
    assert response["ocrReferenceId"] == "req-1"


def test_selected_type_present_among_several_is_processed_normally() -> None:
    raw = parse_extraction_text(
        _extraction("Document_Types_Present: passport, national_id, visa")
    )

    response = normalize_response(
        raw, document_name="Tenant ID", document_filename="bundle.pdf",
        document_type="visa", default_reference_id="req-2",
    )

    assert response["errorInfo"] == []
    assert response["data"]["Passport_Number"] == "C2323323"
    assert response["success"] is True


def test_newly_verifiable_type_is_rejected_when_reported_as_other() -> None:
    raw = parse_extraction_text(_extraction("Document_Types_Present: other"))

    response = normalize_response(
        raw, document_name="Salary", document_filename="salary.pdf",
        document_type="salary_certificate", default_reference_id="req-3",
    )

    assert len(response["errorInfo"]) == 1


def test_named_business_type_is_rejected_when_the_file_is_something_else() -> None:
    # The case that slipped through before every dropdown type became verifiable: Trade License
    # selected, a passport uploaded.
    raw = parse_extraction_text(_extraction("Document_Types_Present: passport"))

    response = normalize_response(
        raw, document_name="Licence", document_filename="passport.pdf",
        document_type="trade_license", default_reference_id="req-5",
    )

    assert response["success"] is False
    assert len(response["errorInfo"]) == 1
    assert "trade licence" in response["errorInfo"][0]["DocumentError"]


def test_type_mismatch_takes_precedence_over_incomplete_document() -> None:
    # Both faults apply; the mismatch is the one worth reporting, since a missing back side of the
    # wrong document is not the caller's actual problem.
    raw = parse_extraction_text(
        "\n".join(["Passport_Number: C2323323", "Front_Side_Visible: true",
                    "Back_Side_Visible: false", "Document_Types_Present: passport"])
    )

    response = normalize_response(
        raw, document_name="Tenant ID", document_filename="passport_scan.jpg",
        document_type="national_id", default_reference_id="req-4",
    )

    assert len(response["errorInfo"]) == 1
    assert "was sent as a National ID, but it looks like a passport" in response["errorInfo"][0]["DocumentError"]


@pytest.mark.parametrize("document_type", ["passport", "visa"])
def test_one_sided_passport_or_visa_is_complete(document_type: str) -> None:
    # Only a National ID needs both sides; a passport or visa showing just its details page is
    # extracted and returned as-is, whatever the model reported about a back side.
    response = normalize_response(
        {"documentComplete": False, "data": {"Tenant_Name_En": "Jane Doe", "Passport_Number": "C2323323"}},
        document_name="Doc",
        document_filename="doc.jpg",
        document_type=document_type,
    )

    assert response["errorInfo"] == []
    assert response["data"]["Passport_Number"] == "C2323323"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Document_Types_Present: cheque", ["cheque"]),
        ("Document_Types_Present: check", ["cheque"]),
        ("Document_Types_Present: security cheque, passport", ["cheque", "passport"]),
        ("Document_Types_Present: salary statement", ["salary_certificate"]),
        ("Document_Types_Present: payslip", ["salary_certificate"]),
    ],
)
def test_cheque_and_salary_spellings_are_recognised(line: str, expected: list[str]) -> None:
    assert parse_extraction_text(_extraction(line))["documentTypesPresent"] == expected


@pytest.mark.parametrize(
    ("requested", "present", "matches"),
    [
        ("cheque", ["cheque"], True),
        ("cheque", ["bank_statement"], False),
        ("cheque", ["other"], False),
        ("salary_certificate", ["salary_certificate"], True),
        ("salary_certificate", ["passport"], False),
    ],
)
def test_cheque_and_salary_certificate_are_type_checked(requested: str, present: list[str], matches: bool) -> None:
    assert document_type_matches(requested, present) is matches
