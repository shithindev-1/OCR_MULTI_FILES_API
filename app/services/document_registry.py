"""The document-type registry: one entry per supported document type (type-detection endpoints).

Once the split step has found a document's type, it is read with that type's own short instruction
and field list instead of the one long instruction for every document. Each entry also says which
details must be visible for the document to count (the covered-parts self-check) and which fields
tie it to a tenant (tenant grouping and the "must belong to the tenant" rule).

The old endpoints (`/ocr`, `/ocr/leasing`) keep the general instruction - see gemini.prompts.
"""

from dataclasses import dataclass

from app.schemas.ocr import PROPERTY_FIELDS

# --- field definitions, reused by every entry that reads the field -------------------------------

_HOLDER = "the document's own holder"

INDIVIDUAL_FIELD_RULES: dict[str, str] = {
    "Property_Area": "Property_Area is the area or community the rented property is in (for example Al Barsha 1, "
                     "Dubai Marina), exactly as printed.",
    "Property_Building": "Property_Building is the rented property's building or project name, exactly as printed.",
    "Property_Plot_Number": "Property_Plot_Number is the property's plot or land number, exactly as printed.",
    "Property_Unit_Number": "Property_Unit_Number is the rented unit's own number (flat, villa or office number), "
                            "exactly as printed - never a plot, makani or contract number.",
    "Property_Contract_Number": "Property_Contract_Number is the Ejari or tenancy contract number the document "
                                "prints for this tenancy, exactly as printed.",
    "Tenant_Name_En": f"Tenant_Name_En is the full name of {_HOLDER} in English, as printed.",
    "Tenant_Name_Ar": f"Tenant_Name_Ar is the full name of {_HOLDER} in Arabic, as printed.",
    "Date_Of_Birth": f"Date_Of_Birth is the date of birth of {_HOLDER}.",
    "Nationality": f"Nationality is the nationality of {_HOLDER}.",
    "Gender": f"Gender is the sex or gender of {_HOLDER}.",
    "Occupation": f"Occupation is the occupation, profession or job title of {_HOLDER}, as printed.",
    "Employer_Name": f"Employer_Name is the employer or sponsor company of {_HOLDER}, as printed with that label.",
    "National_Id": (
        "National_Id is the number of a national identity card - the card itself, or the same card number "
        "printed on a visa or residence permit (for a UAE resident, the Emirates ID number starting 784); it is "
        "never a home-country national or personal number printed on a passport, such as an Egyptian national "
        "number."
    ),
    "National_Id_Expiry": "National_Id_Expiry is the national identity card's own expiry date.",
    "Passport_Number": "Passport_Number is the passport's own number (on a visa: the passport number it prints).",
    "Passport_Issuing_Location": (
        "Passport_Issuing_Location is read only from a passport: the PLACE where it was issued - a city, emirate or "
        "country name printed with a label such as Place of Issue or Issuing Authority. Copy it exactly as printed; "
        "when it is printed only in another language or script (such as Arabic), give its English name instead "
        "(for example \"الشارقة\" -> \"Sharjah\"). An issuing office number or code (for "
        "example \"Issuing Office: 3\") is never a place: when no place name is printed, it is null."
    ),
    "Passport_Expiry": "Passport_Expiry is the passport's own expiry date.",
    "Passport_Issued_Date": "Passport_Issued_Date is the passport's own date of issue.",
    "Visa_Number": (
        "Visa_Number is the visa or residence permit's own number, under whatever label it carries (Visa No, "
        "Visa Number, Control No, Grant Number, Permit No, and so on); a UAE residence visa usually has no field "
        "labeled that, so there read it from \"File\" or \"File Number\" instead."
    ),
    "Visa_Start_Date": "Visa_Start_Date is the visa or residence permit's issue or start date.",
    "Visa_Expiry": "Visa_Expiry is the visa or residence permit's expiry date.",
    "License_Members": (
        "License_Members is every person the document lists - owners, partners, shareholders, managers and "
        "authorised signatories - each written as \"Full Name (Role)\" and separated by semicolons, exactly as printed."
    ),
}

_ADDRESS_FIELDS = (
    "Home_Country", "Home_State_Emirates", "Home_City", "Home_Subdivision_Location",
    "Resident_Country", "Resident_State_Emirates", "Resident_City", "Resident_Subdivision_Location",
    "Street_Name", "Building_No", "Door_No", "PO_Box",
)
_ADDRESS_RULE = (
    "The address fields describe only the holder's OWN address, read only where it is printed as the holder's "
    "address - never a landlord's, employer's, sponsor's, agent's or property owner's address, never the address "
    "of a property being rented unless it is printed as the holder's own address, and never derived from the "
    "place of birth, the place of issue or the nationality. Home_Country, Home_State_Emirates, Home_City and "
    "Home_Subdivision_Location are the holder's permanent address in their home country. Resident_Country, "
    "Resident_State_Emirates, Resident_City, Resident_Subdivision_Location, Street_Name, Building_No, Door_No and "
    "PO_Box are the address where the holder currently lives. In each, State_Emirates is the state, province or "
    "emirate (for example Kerala, or Dubai), City is the city or town, Subdivision_Location is the area, district "
    "or neighbourhood (for example Al Barsha), Building_No is the building name or number, Door_No is the flat, "
    "apartment, villa or unit number, and PO_Box is the P.O. Box number only."
)

COMPANY_FIELD_RULES: dict[str, str] = {
    "Property_Area": "Property_Area is the area or community the rented property is in (for example Al Barsha 1, "
                     "Dubai Marina), exactly as printed.",
    "Property_Building": "Property_Building is the rented property's building or project name, exactly as printed.",
    "Property_Plot_Number": "Property_Plot_Number is the property's plot or land number, exactly as printed.",
    "Property_Unit_Number": "Property_Unit_Number is the rented unit's own number (flat, villa or office number), "
                            "exactly as printed - never a plot, makani or contract number.",
    "Property_Contract_Number": "Property_Contract_Number is the Ejari or tenancy contract number the document "
                                "prints for this tenancy, exactly as printed.",
    "Company_Name_En": "Company_Name_En is the company's registered name in English, as printed.",
    "Company_Name_Ar": "Company_Name_Ar is the company's registered name in Arabic, as printed.",
    "Trade_License_Number": (
        "Trade_License_Number is the trade licence's own licence number; never an initial approval number, "
        "receipt number, register or membership number, or a member's serial number printed on the same page."
    ),
    "Trade_License_Issue_Date": "Trade_License_Issue_Date is the trade licence's own issue date.",
    "Trade_License_Expiry": "Trade_License_Expiry is the trade licence's own expiry date.",
    "Trade_License_Issuer": (
        "Trade_License_Issuer is the authority that issued the licence, for example a Department of Economy, an "
        "economic development department, or a free zone authority, as printed."
    ),
    "Initial_Approval_Number": "Initial_Approval_Number is the initial approval certificate's own approval number.",
    "Initial_Approval_Issue_Date": "Initial_Approval_Issue_Date is its date of approval or issue.",
    "Initial_Approval_Expiry": "Initial_Approval_Expiry is its expiry date.",
    "Initial_Approval_Issuer": "Initial_Approval_Issuer is the authority that issued it.",
    "Signatory_Name_En": "Signatory_Name_En is the card holder's name in English - the holder is the signatory.",
    "Signatory_Name_Ar": "Signatory_Name_Ar is the card holder's name in Arabic.",
    "Signatory_Designation": (
        "Signatory_Designation is the holder's own job title, occupation, profession or role as printed for that "
        "same person - never another person's title, and never a role from a list of several people."
    ),
    "Signatory_National_Id": "Signatory_National_Id is the national identity card's own number.",
    "Signatory_National_Id_Expiry": "Signatory_National_Id_Expiry is the national identity card's own expiry date.",
    "Signatory_Passport_Number": "Signatory_Passport_Number is the passport's own number.",
    "Signatory_Passport_Issuing_Location": (
        "Signatory_Passport_Issuing_Location is read only from the signatory's passport: the PLACE where it was "
        "issued - a city, emirate or country name printed as its place of issue. Copy it exactly as printed; when it "
        "is printed only in another language or script (such as Arabic), give its English name instead (for example "
        "\"الشارقة\" -> \"Sharjah\"). An issuing office number or code is never a place, so then it is null."
    ),
    "Signatory_Passport_Expiry": "Signatory_Passport_Expiry is the passport's own expiry date.",
    "Signatory_Visa_Number": (
        "Signatory_Visa_Number is the visa or residence permit's own number under whatever label it carries; a UAE "
        "residence visa usually has no field labeled that, so there read it from \"File\" or \"File Number\"."
    ),
    "Signatory_Visa_Start_Date": "Signatory_Visa_Start_Date is the visa or residence permit's issue or start date.",
    "Signatory_Visa_Expiry": "Signatory_Visa_Expiry is the visa or residence permit's expiry date.",
    "Holder_Name": "Holder_Name is the full name, in English, of the person the document belongs to, as printed.",
    "License_Members": INDIVIDUAL_FIELD_RULES["License_Members"],
}

_COMMON_RULES = (
    "A value belongs to a field only when it is printed under or beside that field's own label (or, for names, "
    "numbers and dates, in the document's machine-readable lines). A code, office number, serial number or "
    "reference number is never the value of a name, place, nationality or date field - such a field is null "
    "when no real name, place or date is printed. Place fields (nationality, issuing location, country, state "
    "or emirate, city, area) always hold words, never a bare number. When you are not sure a value is meant "
    "for a field, leave it null: an empty field is far better than a wrong one. "
    "For every long number you extract, read it twice, independently, straight off the image both times, "
    "character by character; if the two readings disagree at all, look a third time and use only what you can "
    "actually confirm, and never round a hard-to-read digit to whichever one \"looks about right\". Extract only "
    "values visibly present in the supplied document, each printed with its own label. Never copy a value from "
    "one field into another, never guess, infer, normalize into a value not shown, or use outside knowledge. "
    "Missing, unreadable, or ambiguous values must be null."
)


@dataclass(frozen=True)
class DocumentSpec:
    document_type: str
    label: str                      # how a person names it: "passport", "National ID"
    what: str                       # what the document is and where its details are
    individual_fields: tuple[str, ...]
    company_fields: tuple[str, ...]
    required: str                   # details that must be visible for the document to count
    individual_links: tuple[str, ...] = ()   # numbers that tie it to a person
    company_links: tuple[str, ...] = ()      # numbers that tie it to a person (signatory) or company


_PERSON = ("Tenant_Name_En", "Tenant_Name_Ar", "Date_Of_Birth", "Nationality", "Gender")
_SIGNATORY_IDS = ("Signatory_National_Id", "Signatory_Passport_Number", "Signatory_Visa_Number")
_COMPANY_IDS = ("Trade_License_Number", "Initial_Approval_Number")
_PROPERTY = tuple(PROPERTY_FIELDS)

REGISTRY: dict[str, DocumentSpec] = {spec.document_type: spec for spec in (
    DocumentSpec(
        "national_id", "National ID",
        "a government national identity card from any country, including a UAE Emirates ID. The front and back "
        "may both be in this file: read every field from whichever side prints it (on an Emirates ID the occupation "
        "and employer are on the back).",
        _PERSON + ("Occupation", "Employer_Name", "National_Id", "National_Id_Expiry") + _ADDRESS_FIELDS,
        ("Signatory_Name_En", "Signatory_Name_Ar", "Signatory_Designation", "Signatory_National_Id",
         "Signatory_National_Id_Expiry", "Holder_Name"),
        "the ID number, the holder's name, date of birth, nationality and the expiry date on the front; "
        "the card number and the lines of letters and numbers at the bottom of the back",
        individual_links=("National_Id",), company_links=("Signatory_National_Id",),
    ),
    DocumentSpec(
        "passport", "passport",
        "a passport's data page from any country, possibly with an address or observations page. National_Id is "
        "never read from a passport.",
        _PERSON + ("Occupation", "Passport_Number", "Passport_Issuing_Location", "Passport_Expiry",
                   "Passport_Issued_Date", "Home_Country", "Home_State_Emirates", "Home_City",
                   "Home_Subdivision_Location"),
        ("Signatory_Designation", "Signatory_Passport_Number", "Signatory_Passport_Issuing_Location",
         "Signatory_Passport_Expiry", "Holder_Name"),
        "the passport number, the holder's name, date of birth, nationality, sex, the expiry date and the "
        "two lines of letters and numbers at the bottom of the data page",
        individual_links=("Passport_Number",), company_links=("Signatory_Passport_Number",),
    ),
    DocumentSpec(
        "visa", "visa",
        "a visa or residence permit from any country - a visa page or sticker, an e-visa, or a residence card. It "
        "often prints the holder's passport number and national ID number as well.",
        _PERSON + ("Occupation", "Employer_Name", "National_Id", "Passport_Number", "Visa_Number",
                   "Visa_Start_Date", "Visa_Expiry"),
        ("Signatory_Designation", "Signatory_Visa_Number", "Signatory_Visa_Start_Date", "Signatory_Visa_Expiry",
         "Holder_Name"),
        "the visa or file number, the holder's name and the issue and expiry dates",
        individual_links=("Visa_Number", "Passport_Number", "National_Id"), company_links=("Signatory_Visa_Number",),
    ),
    DocumentSpec(
        "trade_license", "trade licence",
        "a trade licence of any layout or issuer - mainland or free zone, from any emirate or authority, titled "
        "for example Trade Licence, Commercial Licence, Business Licence, Professional Licence or Industrial "
        "Licence, as a card, a certificate or an e-licence. It describes a company, not a person.",
        ("License_Members",),
        ("Company_Name_En", "Company_Name_Ar", "Trade_License_Number", "Trade_License_Issue_Date",
         "Trade_License_Expiry", "Trade_License_Issuer", "License_Members"),
        "the company name, the licence number, the expiry date and the list of owners or managers",
        company_links=("Trade_License_Number",),
    ),
    DocumentSpec(
        "initial_approval", "initial approval certificate",
        "an initial approval certificate or letter - from a government licensing authority for a new business "
        "(such as a Department of Economy) or from a bank or finance company. It is not a trade licence.",
        ("License_Members",),
        ("Company_Name_En", "Company_Name_Ar", "Initial_Approval_Number", "Initial_Approval_Issue_Date",
         "Initial_Approval_Expiry", "Initial_Approval_Issuer", "License_Members"),
        "the approval number, the expiry date and the list of people it names",
        company_links=("Initial_Approval_Number",),
    ),
    DocumentSpec(
        "ejari_certificate", "Ejari certificate",
        "an Ejari tenancy registration certificate. Read the TENANT's details only - never the owner's, the "
        "property manager's or the landlord's.",
        ("Tenant_Name_En", "Tenant_Name_Ar", "Nationality", "National_Id", "Passport_Number", "Visa_Number")
        + _ADDRESS_FIELDS + _PROPERTY,
        ("Company_Name_En", "Company_Name_Ar", "Trade_License_Number") + _PROPERTY,
        "the contract number, the tenant's name and the contract start and end dates",
        individual_links=("National_Id", "Passport_Number", "Visa_Number"), company_links=_COMPANY_IDS,
    ),
    DocumentSpec(
        "tenancy_contract", "tenancy contract",
        "a tenancy or rental contract. Read the TENANT's details only - never the landlord's, owner's or agent's.",
        ("Tenant_Name_En", "Tenant_Name_Ar", "Nationality", "National_Id", "Passport_Number", "Visa_Number")
        + _ADDRESS_FIELDS + _PROPERTY,
        ("Company_Name_En", "Company_Name_Ar", "Trade_License_Number") + _PROPERTY,
        "the tenant's name, the property and the contract start and end dates",
        individual_links=("National_Id", "Passport_Number", "Visa_Number"), company_links=_COMPANY_IDS,
    ),
    DocumentSpec(
        "tenant_form", "tenant form",
        "a tenant registration, application or onboarding form, which may print any of the tenant's details. Read "
        "the tenant's own details only.",
        _PERSON + ("Occupation", "Employer_Name", "National_Id", "National_Id_Expiry", "Passport_Number",
                   "Passport_Expiry", "Visa_Number", "Visa_Expiry") + _ADDRESS_FIELDS,
        ("Company_Name_En", "Company_Name_Ar", "Trade_License_Number"),
        "the tenant's name and the main details filled in",
        individual_links=("National_Id", "Passport_Number", "Visa_Number"), company_links=_COMPANY_IDS,
    ),
    DocumentSpec(
        "salary_certificate", "salary certificate",
        "a salary certificate, salary statement, salary letter or payslip about one employee.",
        ("Tenant_Name_En", "Tenant_Name_Ar", "Nationality", "Occupation", "Employer_Name", "National_Id",
         "Passport_Number"),
        ("Company_Name_En", "Company_Name_Ar"),
        "the employee's name, the employer and the salary",
        individual_links=("National_Id", "Passport_Number"), company_links=(),
    ),
    DocumentSpec(
        "bank_statement", "bank statement",
        "a bank account statement or transaction history. The account holder is the person (or company) whose "
        "details are read.",
        ("Tenant_Name_En", "Tenant_Name_Ar") + _ADDRESS_FIELDS,
        ("Company_Name_En", "Company_Name_Ar"),
        "the account holder's name, the account number and the statement period",
    ),
    DocumentSpec(
        "rental_index", "rental index",
        "a rental index result or certificate from the Dubai Land Department (RERA): it gives the rent range for "
        "a property and describes the property, not a person. In Document_Types_Present, report this document as "
        "rental_index.",
        _PROPERTY,
        _PROPERTY,
        "the area, the property type and the rent range",
    ),
    DocumentSpec(
        "cheque", "cheque",
        "a bank cheque, including a security or post-dated cheque or a copy of one. The account holder (drawer) is "
        "the person (or company) whose details are read - never the payee.",
        ("Tenant_Name_En", "Tenant_Name_Ar"),
        ("Company_Name_En", "Company_Name_Ar"),
        "the account holder's name, the cheque number, the amount and the date",
    ),
)}


def spec(document_type: str) -> DocumentSpec | None:
    return REGISTRY.get(document_type)


def extraction_fields(document_type: str, company: bool) -> list[str] | None:
    """The fields read from a document of this type, or None when the type has no entry."""
    entry = REGISTRY.get(document_type)
    if entry is None:
        return None
    return list(entry.company_fields if company else entry.individual_fields)


def field_rules(document_type: str, company: bool) -> str:
    """The type's own instruction: what the document is, plus the definition of every field it reads."""
    entry = REGISTRY[document_type]
    fields = entry.company_fields if company else entry.individual_fields
    definitions = COMPANY_FIELD_RULES if company else INDIVIDUAL_FIELD_RULES
    parts = [f"This file should hold {_article(entry.label)} {entry.label}: {entry.what}"]
    if not company and any(field in _ADDRESS_FIELDS for field in fields):
        parts.append(_ADDRESS_RULE)
    parts.extend(definitions[field] for field in fields if field in definitions)
    if company and entry.document_type in ("national_id", "passport", "visa"):
        parts.append("This is a document of the person signing for a company tenant; read that person's details.")
    parts.append(_COMMON_RULES)
    return " ".join(parts)


def required_details(document_type: str) -> str:
    entry = REGISTRY.get(document_type)
    return entry.required if entry else "the main identifying details - names, numbers and dates"


def link_numbers(document_type: str, company: bool) -> tuple[str, ...]:
    """The number fields that tie a document of this type to a person (or, for company documents, a
    company). Types without their own entry fall back to every identity number."""
    entry = REGISTRY.get(document_type)
    if entry is None:
        return _SIGNATORY_IDS if company else ("National_Id", "Passport_Number", "Visa_Number")
    return entry.company_links if company else entry.individual_links


# The number a document of this type always prints for itself. A focused reading that misses it (or
# reads nothing at all) is read once more with the general instruction - see ocr_pipeline.
_OWN_NUMBERS = {
    "national_id": ("National_Id", "Signatory_National_Id"),
    "passport": ("Passport_Number", "Signatory_Passport_Number"),
    "visa": ("Visa_Number", "Signatory_Visa_Number"),
    "trade_license": ("Trade_License_Number",),
    "initial_approval": ("Initial_Approval_Number",),
}


def own_numbers(document_type: str, company: bool) -> tuple[str, ...]:
    """The fields holding the document's own number, among the fields it is read with (none for a
    type without one, such as a cheque, or a licence read for an individual)."""
    fields = extraction_fields(document_type, company) or []
    return tuple(field for field in _OWN_NUMBERS.get(document_type, ()) if field in fields)


def _article(label: str) -> str:
    return "an" if label[:1].lower() in "aeiou" else "a"
