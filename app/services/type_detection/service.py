"""Running a type-detection request: split each file into its documents, run them through the
leasing batch with the types found, and build the response."""

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
from app.schemas.leasing import SourceDocument
from app.services import blob_storage
from app.services.concurrency import run_concurrently
from app.services.gemini import get_gemini_service, start_usage_tracking
from app.services.leasing import sources as source_fetch
from app.services.leasing.batch import process_batch
from app.services.leasing.rules import require_mandatory_document
from app.services.leasing.sources import safe_name
from app.services.ocr_pipeline import VerificationUnavailable, call_with_one_retry, log_request_usage
from app.services.post_processing import document_label
from app.services.type_detection.responses import build_detect_response, document_entry
from app.services.type_detection.segmentation import OTHER_TYPE, Segment, parse_side, segment_file
from app.services.validation import ALLOWED_FILES, validate_document_request


def _file_name(item: dict[str, Any]) -> str:
    return item.get("filename") or posixpath.basename(urlparse(item["source"]).path) or item["source"]


async def _segment_one(item: dict[str, Any], request_id: str) -> None:
    """Validates one file and splits it into its documents, in `item["segments"]` - or sets
    `item["error"]`."""
    if item.get("error"):
        return
    upload = SimpleNamespace(filename=item["filename"], content_type=item["content_type"])
    try:
        validated = validate_document_request(
            file=upload, content=item["content"], document_name=safe_name(item["filename"]),
            document_type=OTHER_TYPE, settings=settings,
        )
    except HTTPException as exc:
        item["error"] = f"'{_file_name(item)}': {exc.detail}"
        return
    item["mime_type"] = validated.mime_type
    try:
        item["segments"] = await call_with_one_retry(
            lambda: segment_file(
                content=item["content"], mime_type=validated.mime_type, sha256=validated.sha256,
                settings=settings, gemini_service=get_gemini_service(),
            ),
            what="Segmentation", request_id=request_id,
        )
    except VerificationUnavailable:
        item["error"] = (
            f"We could not read '{_file_name(item)}' right now. Please try again in a few minutes."
        )


def _labels(found: list[tuple[dict[str, Any], Segment]]) -> list[str]:
    """"Passport"; numbered ("National ID 1", "National ID 2") when a type occurs more than once.
    Only the found type - never the file's own name, which may say anything."""
    totals: dict[str, int] = {}
    for _, segment in found:
        totals[segment.document_type] = totals.get(segment.document_type, 0) + 1
    seen: dict[str, int] = {}
    labels = []
    for item, segment in found:
        label = document_label(segment.document_type)
        label = label[:1].upper() + label[1:]
        if totals[segment.document_type] > 1:
            seen[segment.document_type] = seen.get(segment.document_type, 0) + 1
            label = f"{label} {seen[segment.document_type]}"
        labels.append(label)
    return labels


def _unsupported_error(segment: Segment, file_name: str, shared: bool) -> str:
    """The error for a document outside the supported types: what it is, where, and that it is not
    listed - "The DEWA utility bill found on page 2 of 'docs.pdf' is not listed."."""
    name = segment.detected_type
    # "Driving licence" -> "driving licence"; an acronym ("UAE driving licence", "DEWA bill") is kept.
    if name and not (len(name) > 1 and name[1].isupper()):
        name = name[:1].lower() + name[1:]
    where = f"found on {_pages_text(segment.pages)} of '{file_name}'" if shared else f"in '{file_name}'"
    return f"The {name or 'document'} {where} is not listed."


# A side of a document (or a one-sided document) more than this share covered by another document is
# rejected outright - measured in code from the split boxes, so the same photo always gets the same answer.
_MAX_COVERED_SHARE = 0.5


def _covered_error(segment: Segment) -> str | None:
    """The message for a document with a side mostly hidden under another document, or None."""
    for region, share in zip(segment.regions, segment.covered):
        if share > _MAX_COVERED_SHARE:
            kind = document_label(segment.document_type)
            side = parse_side(region.label)
            whose = f"The {side} of your {kind}" if side in ("front", "back") else f"Your {kind}"
            return f"{whose} is mostly covered by another document. Please upload a photo with nothing on top of it."
    return None


def _batch_item(item: dict[str, Any], segment: Segment, label: str, number: int) -> dict[str, Any]:
    """What `process_batch` takes for one found document: the original file when it holds only this
    document, otherwise the document's own cut-out file."""
    document_id = f"{number}_{segment.document_type}"
    batch_item = {
        "source": item["source"], "document_id": document_id,
        "document_type": segment.document_type, "document_name": label,
    }
    if segment.content is None:
        batch_item.update(content=item["content"], filename=item["filename"], content_type=item["content_type"])
        if item.get("blob_path"):
            batch_item["blob_path"] = item["blob_path"]
        return batch_item

    # Already cut to its own edges, with other documents blanked out: the final crop takes it whole.
    batch_item["precut"] = True
    if error := _covered_error(segment):
        batch_item["covered_error"] = error
    stem = Path(item["filename"]).stem or "document"
    filename = f"{safe_name(stem)}_{document_id}{segment.extension}"
    batch_item.update(content=segment.content, filename=filename, content_type=ALLOWED_FILES[segment.extension])
    if item.get("blob_path"):
        # Where its processed copy goes: beside the original, named after its type at upload.
        batch_item["source_blob_path"] = item["blob_path"]
        folder = posixpath.dirname(item["blob_path"])
        batch_item["blob_path"] = posixpath.join(folder, filename) if folder else filename
    return batch_item


def _pages_text(pages: list[int]) -> str:
    if len(pages) == 1:
        return f"page {pages[0]}"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"pages {pages[0]}-{pages[-1]}"
    return "pages " + ", ".join(str(page) for page in pages)


def locate_error(error: str | None, document_type: str, places: list[tuple[str, list[int]]]) -> str | None:
    """Says where a document was found when its file holds several documents, so a message about a
    visa inside "Trade License 11.pdf" reads as being about that visa, not about the licence.

    `places` are (file name, pages) of the sources that sit in a file with other documents; empty
    means every source is a file of its own and the message is returned unchanged. A message starting
    "Your <document>" is reworded to "The <document> found on page N of '<file>'"; any other message
    gets that location in front.
    """
    if not error or not places:
        return error
    plain = document_label(document_type)  # "visa", "National ID", "trade licence"
    label = plain[:1].upper() + plain[1:]
    where = " and ".join(f"{_pages_text(pages)} of '{file_name}'" for file_name, pages in places if pages) or         " and ".join(f"'{file_name}'" for file_name, _ in places)
    lead = f"your {plain}".lower()
    if error.lower().startswith(lead):
        return f"The {plain} found on {where}{error[len(lead):]}"
    return f"{label} found on {where}: {error}"


def _source_row(item: dict[str, Any], pages: list[int], ocr_file: str | None = None) -> dict[str, Any]:
    return {"source": item["source"], "file_name": _file_name(item), "pages": pages, "ocr_file": ocr_file}


async def process_detected(
    *, tenant_type: str, document_refnumber: str, items: list[dict[str, Any]], started_at: float | None = None,
) -> dict[str, Any]:
    """Each item is source, and either content/filename/content_type (plus blob_path when it came
    from blob storage) or the `error` its download failed with."""
    started_at = time.perf_counter() if started_at is None else started_at
    if not items:
        raise HTTPException(status_code=400, detail="Please add at least one document.")

    # One token tally for the whole request, started before the split calls so they are counted
    # too; the batch below adds to it instead of starting a new one.
    start_usage_tracking()
    await run_concurrently([partial(_segment_one, item, document_refnumber) for item in items])

    def finish(status: str, documents: list[dict[str, Any]], tenants: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return build_detect_response(
            status=status, tenant_type=tenant_type, document_refnumber=document_refnumber,
            started_at=started_at, documents=documents, tenants=tenants,
        )

    # A file that could not be read at all leaves nothing to judge the request by.
    if any(item.get("error") for item in items):
        log_request_usage(document_refnumber, "failed", settings.detection_enabled)
        return finish("failed", [
            document_entry(
                document_type="unknown" if item.get("error") else "not_checked",
                status="invalid" if item.get("error") else "not_checked",
                sources=[_source_row(item, [])],
                error=item.get("error"),
            )
            for item in items
        ])

    found = [(item, segment) for item in items for segment in item["segments"]]
    supported = [(item, segment) for item, segment in found if segment.document_type != OTHER_TYPE]
    # Files that hold more than one document: their documents' messages say where each was found.
    shared_files = {id(item) for item in items if len(item["segments"]) > 1}
    # A document outside the supported types is never read: it is rejected, by its name, on its own -
    # it fails the request, not the tenant whose documents are still checked and extracted.
    unsupported = [
        document_entry(
            document_type=OTHER_TYPE, status="invalid", sources=[_source_row(item, segment.pages)],
            error=_unsupported_error(segment, _file_name(item), id(item) in shared_files),
        )
        for item, segment in found if segment.document_type == OTHER_TYPE
    ]

    # Same rule and message as the normal endpoints, judged on the types that were found. Each
    # tenant's own mandatory document is checked once the documents are grouped into tenants.
    try:
        require_mandatory_document(
            tenant_type, [segment.document_type for _, segment in supported], accept_initial_approval=True,
        )
    except HTTPException:
        # Rejected after the split calls were already made - their tokens are still logged.
        log_request_usage(document_refnumber, "rejected", settings.detection_enabled)
        raise

    labels = _labels(supported)
    origin = {label: (item, segment) for label, (item, segment) in zip(labels, supported)}
    batch_items = [
        _batch_item(item, segment, label, number)
        for number, (label, (item, segment)) in enumerate(zip(labels, supported), start=1)
    ]
    result = await process_batch(
        tenant_type=tenant_type, document_refnumber=document_refnumber, items=batch_items,
        started_at=started_at, detected_types=True,
    )

    by_name = result.get("tenant_ids_by_name") or {}
    documents = []
    for report in result["documents"]:
        names = [source["document_name"] for source in report["sources"]]
        places = [
            (_file_name(origin[name][0]), origin[name][1].pages)
            for name in names if id(origin[name][0]) in shared_files
        ]
        documents.append(document_entry(
            tenant_ids=sorted({tenant_id for name in names for tenant_id in by_name.get(name, [])},
                              key=lambda tenant_id: int(tenant_id.rsplit("_", 1)[1])),
            document_type=report["document_type"],
            status="invalid" if report["error"] else "valid",
            merged=report["merged"],
            sources=[
                _source_row(origin[name][0], origin[name][1].pages, source["ocr_file"])
                for name, source in zip(names, report["sources"])
            ],
            error=locate_error(report["error"], report["document_type"], places),
        ))
    status = "failed" if unsupported else result["status"]
    return finish(status, documents + unsupported, tenants=result.get("tenants") or [])


async def process_detected_sources(
    *, tenant_type: str, document_refnumber: str, sources: list[str],
) -> dict[str, Any]:
    """Downloads every link (blob path or allowed URL), then runs `process_detected`."""
    started_at = time.perf_counter()
    storage = blob_storage.get_blob_storage()
    items: list[dict[str, Any]] = [{"source": source} for source in sources]

    async def fetch(client: httpx.AsyncClient, item: dict[str, Any]) -> None:
        name = posixpath.basename(urlparse(item["source"]).path) or "document"
        # The download helpers take a SourceDocument; only its source and name are used.
        source = SourceDocument(source=item["source"], document_id="-", document_type="-", document_name=name)
        blob_path = storage.blob_path(item["source"]) if storage else None
        try:
            if blob_path:
                content, filename, content_type = await source_fetch.download_blob(storage, blob_path, source)
                item["blob_path"] = blob_path
            elif not item["source"].lower().startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail=f"'{item['source']}' is not a valid file link or file path. Please check it and try again.",
                )
            else:
                content, filename, content_type = await source_fetch.download_source(client, source)
        except HTTPException as exc:
            item["error"] = str(exc.detail)
            return
        item.update(content=content, filename=filename, content_type=content_type)

    async with httpx.AsyncClient() as client:
        await run_concurrently([partial(fetch, client, item) for item in items], stage="download")
    return await process_detected(
        tenant_type=tenant_type, document_refnumber=document_refnumber, items=items, started_at=started_at,
    )
