"""Batch processing behind the /ocr/leasing endpoints - see the package docstring for how a
batch is judged."""

import asyncio
import logging
import posixpath
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

from app.core.config import settings
from app.core.exceptions import ValidationStopped
from app.schemas.leasing import LeasingOcrRequest, SourceDocument
from app.schemas.ocr import (
    fields_for_tenant_type,
    is_company,
)
from app.services import blob_storage, progress
from app.services.blob_storage import BlobStorageError
from app.services.concurrency import run_concurrently
from app.services.detection import run_detection, save_merged_document
from app.services.form_normalization import normalize_form_data
from app.services.gemini import get_gemini_service, start_usage_tracking
from app.services.leasing.outputs import (
    document_folder,
    merged_folder,
    output_root,
    upload_ocr_result,
)
from app.services.leasing.pairing import resolve_pairs_per_type
from app.services.leasing.reports import (
    build_response,
    first_error_message,
    fold_merged_pairs,
    merged_indices,
    new_report,
    unconfirmed_type_error,
)
from app.services.leasing.rules import (
    ensure_documents_complete,
    ensure_unique_document_names,
    require_mandatory_document,
)
from app.services.leasing.signatory import apply_signatory_role_from_license
from app.services.leasing.sources import download_blob, download_source, safe_name
from app.services.leasing.tenant_results import finish_per_tenant
from app.services.ocr_pipeline import (
    CROSS_VERIFICATION_UNAVAILABLE,
    VerificationUnavailable,
    extract_and_type_check,
    log_request_usage,
    merge_extracted_data,
    run_self_verification,
    verify_documents_across,
)
from app.services.validation import validate_document_request

logger = logging.getLogger("uae_ocr.leasing")


async def process_batch(
    *, tenant_type: str, document_refnumber: str, items: list[dict[str, Any]],
    started_at: float | None = None, detected_types: bool = False,
) -> dict[str, Any]:
    """Everything from here down is identical for both endpoints - the only difference between
    `/ocr/leasing` and `/ocr/leasing/upload` is how `items` gets built (downloaded vs. read
    straight from the multipart upload) before this function ever runs.

    Each item in `items` is: source (str - the URL, or the uploaded filename), content (bytes),
    filename (str), content_type (str), document_id (str), document_type (str), document_name (str)
    - exactly what a real upload and a downloaded URL both reduce to once the bytes are in hand. An
    item that could not be fetched instead carries `error` (str) and no content.

    Every document is checked in full before anything is decided: validation, extraction and type
    check, front/back pairing, then self-verification. A document that fails a step gets that
    step's message in its own `error` and goes no further; the others carry on. Cross-verification
    runs only when every document passed on its own. Cropping, rotation, the merged PDF and
    `form_data` happen only when the whole request passed.

    Each stage works on its documents concurrently and finishes before the next stage starts, so
    a stage costs as long as its slowest document rather than the sum of all of them. Results are
    always collected back in upload order. A failing document never stops the others from being
    checked; only cross-verification and cropping wait until every document has passed.

    `detected_types` is set by the type-detection endpoint, whose `document_type`s came from the
    segmentation call rather than from the caller. The type check is then strict - extraction's own
    reading must name the same type, and an empty reading does not count as agreement - and its
    message speaks of a type that could not be confirmed rather than one that was sent wrongly.
    """
    started_at = time.perf_counter() if started_at is None else started_at
    request_id = document_refnumber

    if not items:
        raise HTTPException(status_code=400, detail="Please add at least one document.")
    ensure_documents_complete(items, check_source=False)
    require_mandatory_document(tenant_type, [item["document_type"] for item in items])
    fields, _ = fields_for_tenant_type(tenant_type)

    # document_name is assumed unique within one request - it is the only thing the pipeline's own
    # `_merge_front_back` carries forward into a merged document's own `document_name`
    # ("<front name> + <back name>"), so it is the only way this module can trace a result back to
    # the original documents without changing the pipeline itself.
    index_by_name = {item["document_name"]: index for index, item in enumerate(items)}
    documents_report = [new_report(item) for item in items]

    def _finish(status: str, *, form_data: dict[str, Any] | None = None,
                documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return build_response(
            status=status, tenant_type=tenant_type, document_refnumber=document_refnumber,
            started_at=started_at, documents=documents if documents is not None else documents_report,
            form_data=form_data,
        )

    def _failed() -> dict[str, Any]:
        return _finish("failed")

    refnumber_folder = output_root() / safe_name(document_refnumber)
    refnumber_folder.mkdir(parents=True, exist_ok=True)

    start_usage_tracking()

    # One log line per stage: how long it took, and how far into the request it ended.
    stage_started = time.perf_counter()

    def _stage_done(stage: str, documents: int) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        logger.info(
            "Leasing stage=%s request_id=%s documents=%d duration_sec=%.2f elapsed_sec=%.2f",
            stage, request_id, documents, now - stage_started, now - started_at,
        )
        stage_started = now

    progress.report("download")

    # 1. Validation, extraction and type check - every document at once, whatever happens to the
    # others. Returns (candidate or None, whether the OCR provider itself was unavailable).
    async def check_one(index: int, item: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        report = documents_report[index]
        if item.get("error"):
            report["error"] = item["error"]
            return None, False

        position = f"{index + 1}/{len(items)}"
        content, filename, content_type = item["content"], item["filename"], item["content_type"]

        folder = document_folder(document_refnumber, item["document_id"], item["document_name"])
        (folder / f"original_{safe_name(filename)}").write_bytes(content)

        fake_upload = SimpleNamespace(filename=filename, content_type=content_type)
        try:
            validated = validate_document_request(
                file=fake_upload,
                content=content,
                document_name=item["document_name"],
                document_type=item["document_type"],
                settings=settings,
            )
        except HTTPException as exc:
            report["error"] = str(exc.detail)
            return None, False

        report["document_type"] = validated.document_type
        report["sources"][0]["pages"] = list(range(1, (validated.page_count or 1) + 1))

        try:
            candidate = await extract_and_type_check(
                request_id=request_id,
                position=position,
                content=content,
                filename=filename,
                mime_type=validated.mime_type,
                sha256=validated.sha256,
                document_name=item["document_name"],
                document_type=validated.document_type,
                started_at=started_at,
                tenant_type=tenant_type,
                # Detected types get the strict check below instead of the lenient one.
                check_type=not detected_types,
            )
        except HTTPException as exc:
            # The pipeline turns a GeminiServiceError into HTTPException(502, "OCR provider unavailable").
            report["error"] = str(exc.detail)
            return None, True
        except ValidationStopped as stopped:
            # The only stop `_extract_and_type_check` raises is a document type mismatch.
            report["error"] = first_error_message(stopped.response)
            return None, False

        if detected_types and validated.document_type not in candidate["types_present"]:
            report["error"] = unconfirmed_type_error(
                item["document_name"], validated.document_type, candidate["types_present"],
            )
            return None, False

        report["document_type_is_correct"] = True
        candidate["_document_id"] = item["document_id"]
        return candidate, False

    outcomes = await run_concurrently(
        [partial(check_one, index, item) for index, item in enumerate(items)],
        stage="extraction",
    )
    _stage_done("extraction", len(items))
    # Nothing else in the request can be checked without the provider, so stop here.
    if any(provider_down for _, provider_down in outcomes):
        return _failed()
    # Upload order is kept: merge_extracted_data takes the first non-null value in this order.
    candidates = [candidate for candidate, _ in outcomes if candidate is not None]

    # 2. Front/back pairing, per document type.
    resolved = await resolve_pairs_per_type(
        request_id=request_id, candidates=candidates, index_by_name=index_by_name,
        documents_report=documents_report, started_at=started_at, match_several=detected_types,
    )
    _stage_done("pairing", len(candidates))
    progress.report("pairing")

    # 3. Self-verification of every document that got this far, all at once.
    async def verify_one(index: int, document: dict[str, Any]) -> bool:
        try:
            await run_self_verification(
                request_id=request_id,
                position=f"{index}/{len(resolved)}",
                document=document,
                validation_enabled=settings.validation_enabled,
                started_at=started_at,
                short_messages=detected_types,
            )
        except ValidationStopped as stopped:
            message = first_error_message(stopped.response)
            for upload_index in merged_indices(document, index_by_name):
                documents_report[upload_index]["error"] = message
            del document["content"]
            return False
        return True

    passed = await run_concurrently(
        [partial(verify_one, index, document) for index, document in enumerate(resolved, start=1)],
        stage="self_verification",
    )
    extracted = [document for document, ok in zip(resolved, passed) if ok]
    _stage_done("self_verification", len(resolved))

    # Crops, rotates and uploads one passed document's result - used once the request (or, when
    # types were detected, its tenant) has passed.
    async def crop_and_upload(document: dict[str, Any]) -> bool:
        is_merged = bool(document.get("is_merged_pair"))
        upload_indices = merged_indices(document, index_by_name)
        if is_merged:
            front_index, back_index = upload_indices
            output_dir = merged_folder(
                document_refnumber, items[front_index]["document_id"], items[back_index]["document_id"],
            )
        else:
            output_dir = document_folder(document_refnumber, document["_document_id"], document["document_name"])

        saved_paths: list[Path] | None = None
        merged_pdf: Path | None = None
        if settings.detection_enabled:
            per_document_settings = settings.model_copy(update={"detection_output_dir": str(output_dir)})
            saved_paths = await run_detection(
                content=document["content"],
                filename=document["filename"],
                mime_type=document["mime_type"],
                document_type=document["document_type"],
                sha256=document["sha256"],
                settings=per_document_settings,
                gemini_service=get_gemini_service(),
            )
            if is_merged and saved_paths:
                merged_pdf = await asyncio.to_thread(
                    save_merged_document,
                    saved_paths,
                    output_dir=output_dir,
                    filename=document["filename"],
                    sha256=document["sha256"],
                )
        del document["content"]

        # A merged pair uploads one PDF beside the front's original; the back's source row points
        # at that same file.
        blob_path = items[upload_indices[0]].get("blob_path")
        if blob_path and saved_paths:
            try:
                ocr_path = await upload_ocr_result(
                    blob_path, saved_paths, merged_pdf, name=document.get("_ocr_name"),
                )
            except BlobStorageError as exc:
                logger.warning("Processed file upload failed path=%s error=%s", blob_path, exc)
                message = (
                    f"We could not save the processed copy of '{posixpath.basename(blob_path)}'. "
                    "Please try again in a few minutes."
                )
                for index in upload_indices:
                    documents_report[index]["error"] = message
                return False
            for index in upload_indices:
                documents_report[index]["sources"][0]["ocr_file"] = ocr_path
        return True

    if detected_types:
        return await finish_per_tenant(
            tenant_type=tenant_type, request_id=request_id, items=items, candidates=candidates,
            resolved=resolved, extracted=extracted, documents_report=documents_report,
            index_by_name=index_by_name, crop_and_upload=crop_and_upload, finish=_finish,
            stage_done=_stage_done,
        )

    if any(report["error"] for report in documents_report):
        return _failed()

    # 4. Cross-verification - only once every document is valid on its own.
    if len(extracted) > 1 and settings.validation_enabled:
        try:
            issues = await verify_documents_across(
                request_id=request_id, documents=extracted, tenant_type=tenant_type,
                short_messages=detected_types,
            )
        except VerificationUnavailable:
            # The check could not run even after a retry: reject rather than answer "no conflicts".
            # The message names no document, so it lands on every document below.
            issues = [CROSS_VERIFICATION_UNAVAILABLE]
        _stage_done("cross_verification", len(extracted))
        if issues:
            # Each finding names the documents it concerns by their label (document_name). One
            # that names none of them is attached to every document, rather than dropped.
            everyone = list(range(len(items)))
            for issue in issues:
                lowered = issue.lower()
                involved = [
                    index for name, index in index_by_name.items() if name.lower() in lowered
                ] or everyone
                for index in involved:
                    existing = documents_report[index]["error"]
                    if detected_types:
                        # Short messages are whole sentences already; a repeat is dropped.
                        if existing and issue in existing:
                            continue
                        documents_report[index]["error"] = f"{existing} {issue}" if existing else issue
                    else:
                        documents_report[index]["error"] = f"{existing}; {issue}" if existing else issue
            return _failed()
    progress.report("cross_verification")

    # 5. Everything passed - crop, rotate, build merged PDFs and upload _ocr results for every
    # document at once, then fill form_data.
    merged_pairs: list[tuple[int, int, str]] = []
    for document in extracted:
        if document.get("is_merged_pair"):
            front_index, back_index = merged_indices(document, index_by_name)
            merged_pairs.append((front_index, back_index, document["document_type"]))

    await run_concurrently(
        [partial(crop_and_upload, document) for document in extracted],
        stage="crop_rotate_upload",
    )
    _stage_done("crop_rotate_upload", len(extracted))

    if any(report["error"] for report in documents_report):
        return _failed()

    merged_data = merge_extracted_data(extracted, fields)
    if is_company(tenant_type):
        apply_signatory_role_from_license(merged_data, extracted)
    normalize_form_data(merged_data)
    log_request_usage(request_id, "success", settings.detection_enabled)
    return _finish(
        "success", form_data=merged_data,
        documents=fold_merged_pairs(documents_report, merged_pairs),
    )


async def process_sources(request: LeasingOcrRequest) -> dict[str, Any]:
    """POST /ocr/leasing: download every source URL, then run the batch.

    A source that cannot be downloaded gets the reason in its own `error`; every other source is
    still downloaded and checked in full, exactly as if it had been uploaded.
    """
    ensure_documents_complete([source.model_dump() for source in request.sources], check_source=True)
    ensure_unique_document_names([source.document_name for source in request.sources])
    require_mandatory_document(request.tenant_type, [source.document_type for source in request.sources])
    started_at = time.perf_counter()
    items: list[dict[str, Any]] = [
        {
            "source": source.source, "document_id": source.document_id,
            "document_type": source.document_type, "document_name": source.document_name,
        }
        for source in request.sources
    ]
    storage = blob_storage.get_blob_storage()

    async def fetch(client: httpx.AsyncClient, source: SourceDocument, item: dict[str, Any]) -> bool:
        blob_path = storage.blob_path(source.source) if storage else None
        # Where it came from, never the full URL: a URL may carry a SAS token in its query string.
        origin = f"blob:{blob_path}" if blob_path else f"url:{urlparse(source.source).netloc or '?'}"
        fetch_started = time.perf_counter()
        try:
            if blob_path:
                content, filename, content_type = await download_blob(storage, blob_path, source)
                item["blob_path"] = blob_path
            elif not source.source.lower().startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{source.source}' is not a valid file link or file path. Please check it and try again.",
                )
            else:
                content, filename, content_type = await download_source(client, source)
        except HTTPException as exc:
            item["error"] = str(exc.detail)
            logger.warning(
                "Source download failed request_id=%s document_id=%s origin=%s status_code=%d duration_sec=%.2f",
                request.document_refnumber, source.document_id, origin, exc.status_code,
                time.perf_counter() - fetch_started,
            )
            return False
        item.update(content=content, filename=filename, content_type=content_type)
        logger.info(
            "Source downloaded request_id=%s document_id=%s origin=%s bytes=%d duration_sec=%.2f",
            request.document_refnumber, source.document_id, origin, len(content),
            time.perf_counter() - fetch_started,
        )
        return True

    # Every source is downloaded at once; each writes only its own item.
    async with httpx.AsyncClient() as client:
        await run_concurrently(
            [partial(fetch, client, source, item) for source, item in zip(request.sources, items)],
            stage="download",
        )
    logger.info(
        "Leasing stage=download request_id=%s documents=%d duration_sec=%.2f",
        request.document_refnumber, len(items), time.perf_counter() - started_at,
    )

    return await process_batch(
        tenant_type=request.tenant_type, document_refnumber=request.document_refnumber, items=items,
        started_at=started_at,
    )
