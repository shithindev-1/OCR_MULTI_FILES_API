"""Several tenants in one request (type-detection endpoint): grouping the read documents into
tenants, and judging, cropping and filling form_data for each tenant on its own."""

import logging
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any

from app.core.config import settings
from app.schemas.ocr import (
    INDIVIDUAL_REQUIRED_DOCUMENT_TYPE,
    fields_for_tenant_type,
    is_company,
)
from app.services.concurrency import run_concurrently
from app.services.form_normalization import normalize_form_data
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
from app.services.type_detection.tenants import (
    Tenant,
    Unit,
    group_companies,
    group_individuals,
)

logger = logging.getLogger("uae_ocr.leasing")


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
    fields, _ = fields_for_tenant_type(tenant_type)
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

    for key in (key for keys in grouping.unlisted_people for key in keys):
        for index in key:
            add_error(documents_report[index], (
                f"The person on '{items[index]['document_name']}' is not listed on any uploaded trade licence. "
                "Please upload their company's trade licence."
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
                        "None of the people listed on this trade licence were uploaded. "
                        "Please upload the ID documents of a person listed on the licence."
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
    async def cross_check(tenant: Tenant) -> None:
        documents = [passed_by_key[key] for key in tenant.keys if key in passed_by_key]
        if has_errors(tenant) or len(documents) < 2 or not settings.validation_enabled:
            return
        try:
            issues = await verify_documents_across(
                request_id=request_id, documents=documents, tenant_type=tenant_type, short_messages=True,
            )
        except VerificationUnavailable:
            issues = [CROSS_VERIFICATION_UNAVAILABLE]
        own = {items[index]["document_name"]: index for index in indices(tenant)}
        for issue in issues:
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
    # With several tenants, each processed file says whose it is: Tenant_1_National_ID.pdf.
    tenant_numbers = None
    if len(grouping.tenants) > 1:
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
        documents = [passed_by_key[key] for key in sorted(tenant.keys) if key in passed_by_key]
        data = merge_extracted_data(documents, fields)
        if company:
            apply_signatory_role_from_license(data, documents)
        normalize_form_data(data)
        tenants.append({"tenant_id": tenant_id, "status": "success", "form_data": data})

    merged_pairs = [
        (*merged_indices(document, index_by_name), document["document_type"])
        for document in extracted if document.get("is_merged_pair")
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
