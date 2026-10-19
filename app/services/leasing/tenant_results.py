"""Several tenants in one request (type-detection endpoint): grouping the read documents into
tenants, and judging, cropping and filling form_data for each tenant on its own."""

import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from app.core.config import settings
from app.schemas.ocr import (
    INDIVIDUAL_REQUIRED_DOCUMENT_TYPE,
    detect_fields_for_tenant_type,
    is_company,
)
from app.services.concurrency import run_concurrently
from app.services.form_normalization import normalize_form_data
from app.services.leasing.cross_messages import message_for, resolve_documents
from app.services.leasing.outputs import assign_type_file_names
from app.services.leasing.reports import add_error, fold_merged_pairs, merged_indices
from app.services.leasing.signatory import apply_signatory_role_from_license
from app.services.ocr_pipeline import (
    CROSS_VERIFICATION_UNAVAILABLE,
    VerificationUnavailable,
    log_request_usage,
    merge_extracted_data,
    verify_documents_across,
)
from app.services.post_processing import document_label
from app.services.type_detection.tenants import (
    LICENCE_TYPES,
    Tenant,
    Unit,
    group_companies,
    group_individuals,
    has_link_details,
)

logger = logging.getLogger("uae_ocr.leasing")


# Where form_data takes a value from when documents differ: the Emirates ID and visa are current
# and UAE-issued; a passport can be years older (an old occupation, say).
_VALUE_PRIORITY = {"national_id": 0, "visa": 1, "passport": 2}
_ID_FIELDS = ("National_Id", "Signatory_National_Id")


def _without_home_country_id(document: dict[str, Any]) -> dict[str, Any]:
    """A passport's National_Id is its home-country national number (an Egyptian national number,
    say), never the tenant's Emirates ID: it is dropped, so it can neither fill form_data nor be
    reported as a second, different ID number."""
    if document["document_type"] != "passport" or not any(document["data"].get(field) for field in _ID_FIELDS):
        return document
    return {**document, "data": {**document["data"], **{field: None for field in _ID_FIELDS}}}


def _unmatched_message(unit: Unit, name: str, company: bool, *, several: bool) -> str:
    """The error for a supporting document that belongs to no tenant of the request. `name` (its
    label, e.g. "Ejari certificate 2") says which document; the fix names its plain type."""
    whom = "any tenant in this request" if several else "the tenant"
    kind = document_label(unit.document_type)
    if company:
        if has_link_details(unit, company):
            return (f"'{name}' is not for the company on the trade licence or initial approval. "
                    f"Please upload the company's own {kind}, or remove it.")
        return (f"We couldn't confirm that '{name}' is for the company on the trade licence or initial approval. "
                "Please upload a clearer copy.")
    if unit.document_type in LICENCE_TYPES:
        if has_link_details(unit, company):
            return (f"'{name}' doesn't list {whom}. "
                    f"Please upload a {kind} that lists {whom}, or remove it.")
        return f"We couldn't read who is listed on '{name}'. Please upload a clearer copy."
    if has_link_details(unit, company):
        return f"'{name}' doesn't mention {whom}. Please upload the tenant's own {kind}, or remove it."
    return f"We couldn't confirm that '{name}' belongs to {whom}. Please upload a clearer copy."



def _card_aliases(documents: list[dict[str, Any]], own: dict[str, int], types: dict[str, str]) -> dict[str, str]:
    """Makes each front/back pair sent as two files ONE card in `own` and `types` (changed in place),
    and returns the names the model may use for it -> its label.

    The two sides are labelled "National ID 1" and "National ID 2" and the model sees the card as
    "National ID 1 + National ID 2"; it may answer with that, either side, or just "National ID". The
    card is named "National ID" when it is the tenant's only one of its type, otherwise by its front."""
    aliases: dict[str, str] = {}
    for document in documents:
        if not document.get("is_merged_pair"):
            continue
        front, _, back = document["document_name"].partition(" + ")
        if front not in own:
            continue
        index, document_type = own.pop(front), types.pop(front)
        own.pop(back, None)
        types.pop(back, None)
        plain = document_label(document_type)
        plain = plain[:1].upper() + plain[1:]
        label = plain if document_type not in types.values() and plain not in own else front
        own[label], types[label] = index, document_type
        # The plain type name stands for this card only when it is the tenant's one card of that type.
        for name in (document["document_name"], front, back) + ((plain,) if label == plain else ()):
            aliases[name.lower()] = label
    return aliases

async def finish_per_tenant(
    *, tenant_type: str, request_id: str, items: list[dict[str, Any]], candidates: list[dict[str, Any]],
    resolved: list[dict[str, Any]], extracted: list[dict[str, Any]], documents_report: list[dict[str, Any]],
    index_by_name: dict[str, int], crop_and_upload: Callable[[dict[str, Any]], Awaitable[bool]],
    finish: Callable[..., dict[str, Any]], stage_done: Callable[[str, int], None],
) -> dict[str, Any]:
    """The end of `process_batch` for detected types, where one request may hold several tenants.

    Every document that was read is grouped into tenants (app.services.type_detection.tenants). Each tenant is then
    judged on its own: its mandatory document, its documents' own errors, and a cross-check among
    only its documents. A tenant that passes has its documents cropped/uploaded and its own
    form_data filled; a failing tenant's form_data stays empty, and never affects another tenant.

    Returns the normal response plus `tenants` - [{tenant_id, status, form_data}] with form_data in
    the pipeline's field names - and `tenant_ids_by_name` (document_name -> tenant ids).
    """
    company = is_company(tenant_type)
    fields, _ = detect_fields_for_tenant_type(tenant_type)
    passed_by_key = {tuple(merged_indices(document, index_by_name)): document for document in extracted}

    # One unit per document that was read: a merged front/back pair is one unit of both uploads.
    units: list[Unit] = []
    covered: set[int] = set()
    for document in resolved:
        key = tuple(merged_indices(document, index_by_name))
        covered.update(key)
        units.append(Unit(key=key, document_type=document["document_type"], data=document["data"]))
    for candidate in candidates:
        index = index_by_name[candidate["document_name"]]
        if index not in covered:  # read, but failed front/back pairing
            units.append(Unit(key=(index,), document_type=candidate["document_type"], data=candidate["data"]))
    units.sort(key=lambda unit: unit.key[0])

    grouping = group_companies(units) if company else group_individuals(units)
    tenant_ids = [f"tenant_{number}" for number in range(1, len(grouping.tenants) + 1)]
    # Upload positions (1-based) and types only - never names or numbers read off the documents.
    logger.info(
        "Tenant grouping request_id=%s tenant_type=%s tenants=%s unassigned=%s unlisted_people=%s",
        request_id, "company" if company else "individual",
        {
            tenant_id: [f"{index + 1}:{items[index]['document_type']}" for key in tenant.keys for index in key]
            for tenant_id, tenant in zip(tenant_ids, grouping.tenants)
        },
        [index + 1 for key in grouping.unassigned for index in key],
        [[index + 1 for key in keys for index in key] for keys in grouping.unlisted_people],
    )
    tenant_ids_by_index: dict[int, list[str]] = {}
    for tenant_id, tenant in zip(tenant_ids, grouping.tenants):
        for key in tenant.keys:
            for index in key:
                tenant_ids_by_index.setdefault(index, []).append(tenant_id)

    # A supporting document that belongs to no tenant is reported on itself. It joins no tenant, so
    # it never fails one: the tenants' own documents are still checked and extracted, it gets no
    # processed file, and it adds nothing to form_data - only the request's status says to fix it.
    unit_by_key = {unit.key: unit for unit in units}
    for key in grouping.unassigned:
        for index in key:
            add_error(documents_report[index], _unmatched_message(
                unit_by_key[key], items[index]["document_name"], company, several=len(grouping.tenants) > 1,
            ))

    for key in (key for keys in grouping.unlisted_people for key in keys):
        for index in key:
            add_error(documents_report[index], (
                f"The person on '{items[index]['document_name']}' is not listed on any uploaded trade licence "
                "or initial approval. Please upload their company's trade licence or initial approval."
            ))

    def indices(tenant: Tenant) -> list[int]:
        return sorted({index for key in tenant.keys for index in key})

    def has_errors(tenant: Tenant) -> bool:
        return any(documents_report[index]["error"] for index in indices(tenant))

    # A company needs a signatory: someone its licence lists, whose own documents were uploaded. A
    # licence nobody uploaded was signed for fails, rather than passing with company details alone.
    if company:
        for tenant in grouping.tenants:
            if not tenant.signatory_keys:
                for index in indices(tenant):
                    add_error(documents_report[index], (
                        "None of the people listed on this trade licence or initial approval were uploaded. "
                        "Please upload the ID documents of a person it lists."
                    ))

    # Each individual needs their own National ID.
    if not company:
        for tenant in grouping.tenants:
            if not any(items[index]["document_type"] == INDIVIDUAL_REQUIRED_DOCUMENT_TYPE for index in indices(tenant)):
                names = [items[index]["document_name"] for index in indices(tenant)]
                for index in indices(tenant):
                    add_error(documents_report[index], (
                        f"No National ID was uploaded for the person on '{names[0]}'. Please upload their National ID."
                    ))

    # Cross-check each tenant's own documents, all tenants at once.
    def own_documents(tenant: Tenant) -> list[dict[str, Any]]:
        """The tenant's passed documents whose fields describe the tenant - for an individual, not a
        licence that merely lists them - in the order form_data takes values from: the National ID
        first, then the visa, then the passport, then the rest (the first value found wins)."""
        documents = [
            _without_home_country_id(passed_by_key[key]) for key in sorted(tenant.keys)
            if key in passed_by_key and (company or passed_by_key[key]["document_type"] not in LICENCE_TYPES)
        ]
        return sorted(documents, key=lambda document: _VALUE_PRIORITY.get(document["document_type"], len(_VALUE_PRIORITY)))

    async def cross_check(tenant: Tenant) -> None:
        documents = own_documents(tenant)
        if has_errors(tenant) or len(documents) < 2 or not settings.validation_enabled:
            return
        try:
            issues = await verify_documents_across(
                request_id=request_id, documents=documents, tenant_type=tenant_type, short_messages=True,
            )
        except VerificationUnavailable:
            issues = [CROSS_VERIFICATION_UNAVAILABLE]
        own = {items[index]["document_name"]: index for index in indices(tenant)}
        types = {name: items[index]["document_type"] for name, index in own.items()}
        aliases = _card_aliases(documents, own, types)
        for issue in issues:
            if isinstance(issue, dict):
                # A structured conflict: each involved document gets its own message.
                named = resolve_documents(issue.get("documents") or [], types, aliases)
                if not named:
                    named = list(own)  # never drop a finding: it lands on every document
                for label in named:
                    add_error(documents_report[own[label]], message_for(label, {**issue, "documents": named}, types))
                continue
            # Labels are plain type names ("Passport"), which can also appear as ordinary words in
            # a message, so a quoted label - as the prompt asks for - is matched first.
            lowered = issue.lower()
            involved = (
                [index for name, index in own.items() if f"'{name.lower()}'" in lowered]
                or [index for name, index in own.items() if name.lower() in lowered]
                or list(own.values())
            )
            for index in involved:
                add_error(documents_report[index], issue)

    await run_concurrently([partial(cross_check, tenant) for tenant in grouping.tenants], stage="cross_verification")
    stage_done("cross_verification", len(extracted))

    # Crop and upload every document of every passing tenant - once, even when shared.
    to_crop = {
        key: passed_by_key[key]
        for tenant in grouping.tenants if not has_errors(tenant)
        for key in tenant.keys if key in passed_by_key
    }
    # Every processed file says whose it is - also with a single tenant: Tenant_1_National_ID.pdf.
    tenant_numbers = {
        key: [number for number, tenant in enumerate(grouping.tenants, start=1) if key in tenant.keys]
        for key in to_crop
    }
    assign_type_file_names(to_crop, items, tenant_numbers)
    await run_concurrently([partial(crop_and_upload, document) for document in to_crop.values()], stage="crop_rotate_upload")
    stage_done("crop_rotate_upload", len(to_crop))
    for document in extracted:
        document.pop("content", None)

    tenants = []
    for tenant_id, tenant in zip(tenant_ids, grouping.tenants):
        if has_errors(tenant):
            tenants.append({"tenant_id": tenant_id, "status": "failed", "form_data": {}})
            continue
        documents = own_documents(tenant)
        data = merge_extracted_data(documents, fields)
        if company:
            apply_signatory_role_from_license(data, documents)
        normalize_form_data(data)
        tenants.append({"tenant_id": tenant_id, "status": "success", "form_data": data})

    # Every front/back pair read as one card is one entry - also one that failed, so its message
    # is shown once, and a pair failed by a later check keeps that check's message.
    merged_pairs = [
        (*merged_indices(document, index_by_name), document["document_type"])
        for document in resolved if document.get("is_merged_pair")
    ]
    status = "success" if tenants and not any(report["error"] for report in documents_report) else "failed"
    logger.info(
        "Tenant results request_id=%s status=%s tenants=%s",
        request_id, status, {tenant["tenant_id"]: tenant["status"] for tenant in tenants},
    )
    log_request_usage(request_id, status, settings.detection_enabled)
    response = finish(status, documents=fold_merged_pairs(documents_report, merged_pairs))
    response["tenants"] = tenants
    response["tenant_ids_by_name"] = {
        items[index]["document_name"]: ids for index, ids in tenant_ids_by_index.items()
    }
    return response
