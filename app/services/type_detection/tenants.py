"""Grouping one request's documents into tenants - for the type-detection endpoint.

A request may carry several tenants' documents together. Once every document has been read, they
are grouped here, deterministically, from what was read off them:

  - Individual: one tenant per person. Two identity documents are the same person when they share
    an ID number (National ID, passport or visa number - a visa routinely prints the passport
    number), or have the same date of birth and at least one name in common, or the same name with
    no conflicting date of birth.
  - Company: one company per trade licence (two copies of one licence - same number or name - are
    one company). Each person's identity documents are grouped as above, then attached to every
    company whose licence lists that person. A company becomes one tenant per attached person, or
    one tenant with no signatory when nobody it lists was uploaded - which then fails, as a company
    needs a listed signatory (see app.services.leasing.tenant_results).

Supporting documents (Ejari, tenancy contract, tenant form, ...) join the tenant whose ID number or
name they print; with only one tenant they join it; otherwise they stay unassigned.

Each unit is one document: a key (anything hashable the caller uses to find it again), its type,
and its extracted fields.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Hashable

from app.services.form_normalization import format_date
from app.services.names import name_tokens, same_name

PERSON_TYPES = frozenset({"national_id", "passport", "visa"})
COMPANY_TYPE = "trade_license"

_INDIVIDUAL_IDS = ("National_Id", "Passport_Number", "Visa_Number")
_SIGNATORY_IDS = ("Signatory_National_Id", "Signatory_Passport_Number", "Signatory_Visa_Number")
_MEMBER = re.compile(r"^\s*(?P<name>[^()]+?)\s*(?:\([^()]*\))?\s*$")


@dataclass
class Unit:
    key: Hashable
    document_type: str
    data: dict[str, Any]


@dataclass
class Tenant:
    """One form_data entry: the documents it is built from."""

    keys: list[Hashable] = field(default_factory=list)
    # For a company: the licence's documents, and the signatory's (empty when nobody it lists was uploaded).
    company_keys: list[Hashable] = field(default_factory=list)
    signatory_keys: list[Hashable] = field(default_factory=list)


@dataclass
class Grouping:
    tenants: list[Tenant]
    # Supporting documents that could not be tied to one tenant.
    unassigned: list[Hashable]
    # Company only: people whose documents name no person on any uploaded licence.
    unlisted_people: list[list[Hashable]]


def _id(value: Any) -> str | None:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    return text if len(text) >= 5 else None


def _birth(value: Any) -> str | None:
    return format_date(str(value), birth=True) if value else None


@dataclass
class _Person:
    keys: list[Hashable] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    names: list[str] = field(default_factory=list)
    births: set[str] = field(default_factory=set)

    def matches(self, ids: set[str], name: str | None, birth: str | None) -> bool:
        if ids & self.ids:
            return True
        if birth and birth in self.births and name and any(name_tokens(name) & name_tokens(known) for known in self.names):
            return True
        if name and any(same_name(name, known) for known in self.names):
            return not (birth and self.births and birth not in self.births)
        return False

    def add(self, key: Hashable, ids: set[str], name: str | None, birth: str | None) -> None:
        self.keys.append(key)
        self.ids |= ids
        if name:
            self.names.append(name)
        if birth:
            self.births.add(birth)


def _person_details(unit: Unit, company: bool) -> tuple[set[str], str | None, str | None]:
    data = unit.data
    if company:
        ids = {_id(data.get(name)) for name in _SIGNATORY_IDS}
        name = data.get("Signatory_Name_En") or data.get("Holder_Name")
        birth = None
    else:
        ids = {_id(data.get(name)) for name in _INDIVIDUAL_IDS}
        name, birth = data.get("Tenant_Name_En"), _birth(data.get("Date_Of_Birth"))
    return {value for value in ids if value}, name, birth


def _group_people(units: list[Unit], company: bool) -> list[_Person]:
    people: list[_Person] = []
    for unit in units:
        ids, name, birth = _person_details(unit, company)
        matching = [person for person in people if person.matches(ids, name, birth)]
        if not matching:
            person = _Person()
            people.append(person)
        else:
            # One document can link two groups found so far (a visa printing a passport number
            # already seen alongside an ID card's name): they are one person.
            person = matching[0]
            for other in matching[1:]:
                person.keys += other.keys
                person.ids |= other.ids
                person.names += other.names
                person.births |= other.births
                people.remove(other)
        person.add(unit.key, ids, name, birth)
    return people


def _identity(unit: Unit, company: bool) -> tuple[set[str], str | None]:
    """What a supporting document is matched on: the company's licence number and name for a
    company, the person's ID numbers and name for an individual."""
    if company:
        number = _id(unit.data.get("Trade_License_Number"))
        return ({number} if number else set()), unit.data.get("Company_Name_En")
    ids, name, _ = _person_details(unit, company=False)
    return ids, name


def _attach_supporting(
    supporting: list[Unit], tenants: list[Tenant], unit_by_key: dict[Hashable, Unit], company: bool,
) -> list[Hashable]:
    """Adds each supporting document to the tenant(s) it names; returns those that fit none. A
    company document joins every tenant of that company (one per signatory)."""
    unassigned: list[Hashable] = []
    for unit in supporting:
        if len(tenants) == 1:
            tenants[0].keys.append(unit.key)
            continue
        ids, name = _identity(unit, company)

        def names_it(tenant: Tenant) -> bool:
            for key in tenant.company_keys if company else tenant.keys:
                own_ids, own_name = _identity(unit_by_key[key], company)
                if (ids & own_ids) or (name and own_name and same_name(name, own_name)):
                    return True
            return False

        owners = [tenant for tenant in tenants if names_it(tenant)]
        if not owners:
            unassigned.append(unit.key)
        for tenant in owners if company else owners[:1]:
            tenant.keys.append(unit.key)
    return unassigned


def group_individuals(units: list[Unit]) -> Grouping:
    unit_by_key = {unit.key: unit for unit in units}
    people = _group_people([unit for unit in units if unit.document_type in PERSON_TYPES], company=False)
    tenants = [Tenant(keys=list(person.keys)) for person in people]
    supporting = [unit for unit in units if unit.document_type not in PERSON_TYPES]
    unassigned = _attach_supporting(supporting, tenants, unit_by_key, company=False)
    return Grouping(tenants=tenants, unassigned=unassigned, unlisted_people=[])


def _members(unit: Unit) -> list[str]:
    names = []
    for member in str(unit.data.get("License_Members") or "").split(";"):
        match = _MEMBER.match(member)
        if match and match["name"].strip():
            names.append(match["name"].strip())
    return names


def group_companies(units: list[Unit]) -> Grouping:
    unit_by_key = {unit.key: unit for unit in units}

    companies: list[list[Unit]] = []
    for unit in (unit for unit in units if unit.document_type == COMPANY_TYPE):
        number, name = _id(unit.data.get("Trade_License_Number")), unit.data.get("Company_Name_En")
        for company in companies:
            if any(
                (number and number == _id(other.data.get("Trade_License_Number")))
                or (name and same_name(name, other.data.get("Company_Name_En")))
                for other in company
            ):
                company.append(unit)
                break
        else:
            companies.append([unit])

    people = _group_people([unit for unit in units if unit.document_type in PERSON_TYPES], company=True)
    tenants: list[Tenant] = []
    listed: set[int] = set()
    for company in companies:
        members = [member for unit in company for member in _members(unit)]
        company_keys = [unit.key for unit in company]
        attached = [
            index for index, person in enumerate(people)
            if any(same_name(name, member) for name in person.names for member in members)
        ]
        listed.update(attached)
        if not attached:
            tenants.append(Tenant(keys=list(company_keys), company_keys=company_keys))
        for index in attached:
            tenants.append(Tenant(
                keys=company_keys + people[index].keys, company_keys=company_keys,
                signatory_keys=list(people[index].keys),
            ))
    unlisted = [person.keys for index, person in enumerate(people) if index not in listed]

    supporting = [unit for unit in units if unit.document_type not in PERSON_TYPES and unit.document_type != COMPANY_TYPE]
    unassigned = _attach_supporting(supporting, tenants, unit_by_key, company=True)
    return Grouping(tenants=tenants, unassigned=unassigned, unlisted_people=unlisted)
