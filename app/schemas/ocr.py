"""Field lists and response models for the /ocr endpoint."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

DATA_FIELDS = [
    "Tenant_Name_En",
    "Tenant_Name_Ar",
    "Date_Of_Birth",
    "Nationality",
    "Gender",
    "Occupation",
    "Employer_Name",
    "National_Id",
    "National_Id_Expiry",
    "Passport_Number",
    "Passport_Issuing_Location",
    "Passport_Expiry",
    "Visa_Number",
    "Visa_Start_Date",
    "Visa_Expiry",
    "Home_Country",
    "Home_State_Emirates",
    "Home_City",
    "Home_Subdivision_Location",
    "Resident_Country",
    "Resident_State_Emirates",
    "Resident_City",
    "Resident_Subdivision_Location",
    "Street_Name",
    "Building_No",
    "Door_No",
    "PO_Box",
    "Passport_Issued_Date",
]

# Only a National ID carries a mandatory front/back pair. Every other document type - passport and
# visa included - is complete as submitted, so a single-page upload of one never raises an
# incomplete error.
FRONT_BACK_TYPES = frozenset({"national_id"})
# What the type-detection split calls a document outside the supported types.
OTHER_DOCUMENT_TYPE = "other"
# Types the type-detection endpoints recognise on top of VERIFIABLE_TYPES. The older endpoints keep their
# own list: a caller there declares the type, and their prompts are unchanged.
DETECT_ONLY_TYPES = frozenset({"rental_index"})
# Property details read (never returned in form_data) to match a rental index to the tenant's Ejari or
# tenancy contract - type-detection endpoints only.
PROPERTY_FIELDS = [
    "Property_Area", "Property_Building", "Property_Plot_Number", "Property_Unit_Number", "Property_Contract_Number",
]

# Every type the extraction pass is taught to recognise by name, and therefore every type whose
# selection can be checked against what the file actually shows. These are exactly the values the
# client's document-type choices send. A caller may still send some other string - "utility_bill", say -
# which the model has no name for; those are accepted as declared rather than guessed at, because
# rejecting on an observation the model was never asked to make would fail valid uploads.
VERIFIABLE_TYPES = frozenset({
    "national_id",
    "passport",
    "visa",
    "bank_statement",
    "ejari_certificate",
    "trade_license",
    "tenant_form",
    "initial_approval",
    "salary_certificate",
    "tenancy_contract",
    "cheque",
})
# Every type the type-detection endpoints recognise.
DETECT_DOCUMENT_TYPES = VERIFIABLE_TYPES | DETECT_ONLY_TYPES

# Other spellings of the VERIFIABLE_TYPES, keyed by their normalised form (lowercase, underscores).
# Applied both to the type a caller declares and to the types the model reports, so "Salary
# Statement", "Emirates ID" or "Cheque Copy" are checked as salary_certificate, national_id and
# cheque rather than slipping through as unknown types.
DOCUMENT_TYPE_ALIASES = {
    "nationalid": "national_id",
    "national_identity": "national_id",
    "emirates_id": "national_id",
    "eid": "national_id",
    "residence_visa": "visa",
    "residency_visa": "visa",
    # British spelling of the same document, which the model uses about as often as the American.
    "trade_licence": "trade_license",
    "business_license": "trade_license",
    "business_licence": "trade_license",
    "ejari": "ejari_certificate",
    "ejari_registration": "ejari_certificate",
    "bank_account_statement": "bank_statement",
    "statement_of_account": "bank_statement",
    "initialapproval": "initial_approval",
    "initial_approval_letter": "initial_approval",
    "rental_index_result": "rental_index",
    "rental_index_certificate": "rental_index",
    "rera_rental_index": "rental_index",
    "dld_rental_index": "rental_index",
    "salary_certificate_letter": "salary_certificate",
    "salary_cert": "salary_certificate",
    "salary_statement": "salary_certificate",
    "salary_letter": "salary_certificate",
    "salary_slip": "salary_certificate",
    "salary_transfer_letter": "salary_certificate",
    "payslip": "salary_certificate",
    "pay_slip": "salary_certificate",
    "tenancy_agreement": "tenancy_contract",
    "rental_contract": "tenancy_contract",
    "check": "cheque",
    "cheques": "cheque",
    "bank_cheque": "cheque",
    "cheque_copy": "cheque",
    "security_cheque": "cheque",
    "post_dated_cheque": "cheque",
    "postdated_cheque": "cheque",
    "pdc": "cheque",
}


def canonical_document_type(document_type: str) -> str:
    """The one spelling a type is checked under, e.g. "salary_statement" -> "salary_certificate"."""
    return DOCUMENT_TYPE_ALIASES.get(document_type, document_type)

MISSING_INFO_FIELDS = [
    "tenant_name_en",
    "tenant_name_ar",
    "date_of_birth",
    "nationality",
    "gender",
    "occupation",
    "employer_name",
    "national_id",
    "national_id_expiry",
    "passport_number",
    "passport_issuing_location",
    "passport_expiry",
    "visa_number",
    "visa_start_date",
    "visa_expiry",
    "home_country",
    "home_state_emirates",
    "home_city",
    "home_subdivision_location",
    "resident_country",
    "resident_state_emirates",
    "resident_city",
    "resident_subdivision_location",
    "street_name",
    "building_no",
    "door_no",
    "po_box",
    "passport_issued_date",
]


# tenant_type=company extracts these instead of the individual fields above: the company itself,
# from its trade licence and other company documents, and the person signing for it, from that
# signatory's own ID, passport and visa. Same pairing as DATA_FIELDS -> MISSING_INFO_FIELDS.
COMPANY_DATA_FIELDS = [
    "Company_Name_En",
    "Company_Name_Ar",
    "Trade_License_Number",
    "Trade_License_Issue_Date",
    "Trade_License_Expiry",
    "Trade_License_Issuer",
    "Signatory_Name_En",
    "Signatory_Name_Ar",
    "Signatory_Designation",
    "Signatory_National_Id",
    "Signatory_National_Id_Expiry",
    "Signatory_Passport_Number",
    "Signatory_Passport_Issuing_Location",
    "Signatory_Passport_Expiry",
    "Signatory_Visa_Number",
    "Signatory_Visa_Start_Date",
    "Signatory_Visa_Expiry",
]

COMPANY_MISSING_INFO_FIELDS = [field.lower() for field in COMPANY_DATA_FIELDS]

# Read from company documents for the cross-check only - never returned in form_data or missing.
# License_Members: everyone a trade licence lists, so the signatory can be checked against them.
# Holder_Name: whose national ID, passport or visa a document is - the signatory name in form_data
# comes only from the national ID, so this is how the other personal documents are matched to them.
COMPANY_INTERNAL_FIELDS = ["License_Members", "Holder_Name"]
# An initial approval certificate's own details - kept apart from the trade licence's. Read for every
# company request, returned only by the type-detection endpoints (the /ocr/leasing response is unchanged).
INITIAL_APPROVAL_FIELDS = [
    "Initial_Approval_Number",
    "Initial_Approval_Issue_Date",
    "Initial_Approval_Expiry",
    "Initial_Approval_Issuer",
]
# What the company extraction prompt asks for: the output fields, the initial approval's own fields,
# and the internal ones.
COMPANY_EXTRACTION_FIELDS = COMPANY_DATA_FIELDS + INITIAL_APPROVAL_FIELDS + COMPANY_INTERNAL_FIELDS
# A company's form_data on the type-detection endpoints: the trade licence fields plus the initial approval's.
DETECT_COMPANY_DATA_FIELDS = COMPANY_DATA_FIELDS + INITIAL_APPROVAL_FIELDS
DETECT_COMPANY_MISSING_INFO_FIELDS = [field.lower() for field in DETECT_COMPANY_DATA_FIELDS]

# Read from an individual's documents for matching only - never returned in form_data or missing.
# License_Members: everyone a trade licence or initial approval lists, so such a document can be
# accepted only when it lists the tenant.
INDIVIDUAL_INTERNAL_FIELDS = ["License_Members"]
# What the individual extraction prompt asks for: the output fields plus the internal ones.
INDIVIDUAL_EXTRACTION_FIELDS = DATA_FIELDS + INDIVIDUAL_INTERNAL_FIELDS

TENANT_TYPE_COMPANY = "company"
# The one document a company submission cannot go without.
COMPANY_REQUIRED_DOCUMENT_TYPE = "trade_license"
# On the type-detection endpoints, an initial approval satisfies that rule too.
COMPANY_ALTERNATIVE_DOCUMENT_TYPE = "initial_approval"
# The one document an individual submission cannot go without.
INDIVIDUAL_REQUIRED_DOCUMENT_TYPE = "national_id"


def is_company(tenant_type: str | None) -> bool:
    return (tenant_type or "").strip().lower() == TENANT_TYPE_COMPANY


def fields_for_tenant_type(tenant_type: str | None) -> tuple[list[str], list[str]]:
    """(extraction field names, form_data/missing key names) for a tenant type."""
    if is_company(tenant_type):
        return COMPANY_DATA_FIELDS, COMPANY_MISSING_INFO_FIELDS
    return DATA_FIELDS, MISSING_INFO_FIELDS


def detect_fields_for_tenant_type(tenant_type: str | None) -> tuple[list[str], list[str]]:
    """Like `fields_for_tenant_type`, for the type-detection endpoints: a company also gets the
    initial approval's own keys."""
    if is_company(tenant_type):
        return DETECT_COMPANY_DATA_FIELDS, DETECT_COMPANY_MISSING_INFO_FIELDS
    return DATA_FIELDS, MISSING_INFO_FIELDS


class OcrData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Tenant_Name_En: str | None = None
    Tenant_Name_Ar: str | None = None
    Date_Of_Birth: str | None = None
    Nationality: str | None = None
    Gender: str | None = None
    Occupation: str | None = None
    Employer_Name: str | None = None
    National_Id: str | None = None
    National_Id_Expiry: str | None = None
    Passport_Number: str | None = None
    Passport_Issuing_Location: str | None = None
    Passport_Expiry: str | None = None
    Visa_Number: str | None = None
    Visa_Start_Date: str | None = None
    Visa_Expiry: str | None = None
    Home_Country: str | None = None
    Home_State_Emirates: str | None = None
    Home_City: str | None = None
    Home_Subdivision_Location: str | None = None
    Resident_Country: str | None = None
    Resident_State_Emirates: str | None = None
    Resident_City: str | None = None
    Resident_Subdivision_Location: str | None = None
    Street_Name: str | None = None
    Building_No: str | None = None
    Door_No: str | None = None
    PO_Box: str | None = None
    Passport_Issued_Date: str | None = None


class ErrorInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    DocumentName: str
    DocumentFileName: str
    DocumentError: str
    DocumentErrorToShow: str


class OcrResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool = True
    userName: str | None = None
    tenantType: str | None = None
    subscriptionId: str | None = None
    ocrReferenceId: str | None = None
    data: OcrData = Field(default_factory=OcrData)
    missingInfo: list[str] = Field(default_factory=list)
    errorInfo: list[ErrorInfo] = Field(default_factory=list)

    @model_validator(mode="after")
    def _derive_success(self) -> "OcrResponse":
        # success reflects whether any field was actually extracted, not whether the
        # request completed without a transport-level error (that's the HTTP status code's
        # job) — a document that yields nothing usable must report success=false even
        # though the API call itself succeeded.
        self.success = any(value is not None for value in self.data.model_dump().values())
        return self


def empty_response() -> dict[str, Any]:
    return OcrResponse().model_dump()
