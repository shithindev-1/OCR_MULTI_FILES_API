"""The leasing endpoints that find each document's type themselves.

  - `POST /ocr/leasing/detect`              JSON: tenant_type, document_refnumber, sources[{source}].
  - `POST /ocr/leasing/detect/upload`       form-data: tenant_type, document_refnumber, and repeated
                                            `file` fields - or `sources[n][source]` links.
  - `POST /ocr/leasing/detect/jobs`         same JSON as /detect, run in the background -> {"job_id"}
  - `POST /ocr/leasing/detect/jobs/upload`  same form-data as /detect/upload, in the background.

A job is polled with the normal `GET /ocr/leasing/jobs/{job_id}`; its `result` is the same response
the matching direct endpoint gives.

No document_type, document_name or document_id is taken: a request carrying any of them is
rejected rather than silently ignored, so a caller never believes a declared type was used. All
return `DetectOcrResponse`. See app.services.type_detection for how a request is judged.
"""

import time
from functools import partial
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from app.api.forms import Work, form_sources
from app.schemas.detect import DetectOcrRequest, DetectOcrResponse
from app.services import jobs
from app.services.type_detection.responses import failed_detect_response
from app.services.type_detection.service import (
    process_detected,
    process_detected_sources,
)

router = APIRouter(prefix="/ocr/leasing/detect", tags=["leasing - type detection"])

_DECLARED_FIELDS = ("document_type", "document_name", "document_id")


def _check_links(links: list[str]) -> None:
    """Rejects an empty list or an empty link at once - before any job is created."""
    if not links:
        raise HTTPException(status_code=400, detail="Please add at least one document.")
    for position, link in enumerate(links, start=1):
        if not link.strip():
            raise HTTPException(status_code=422, detail=f"Source {position} is empty. Please add a file link.")


def _sources_work(request: DetectOcrRequest) -> tuple[Work, list[str]]:
    sources = [source.source for source in request.sources]
    _check_links(sources)
    work = partial(
        process_detected_sources,
        tenant_type=request.tenant_type, document_refnumber=request.document_refnumber, sources=sources,
    )
    return work, sources


async def _upload_work(
    request: Request, tenant_type: str, document_refnumber: str, file: list[UploadFile] | None,
) -> tuple[Work, list[str]]:
    """Checks a form-data request and reads its files. Returns the processing still to run - which
    can also run in the background after the response is sent - and the name of each file or link."""
    form = await request.form()
    sources = form_sources(form)
    declared = sorted(
        {key for key in form.keys() if key in _DECLARED_FIELDS}
        | {field for row in sources for field in row if field in _DECLARED_FIELDS}
    )
    if declared:
        raise HTTPException(
            status_code=400,
            detail=f"This endpoint finds each document's type itself - please do not send {', '.join(declared)}. "
                   "Send only the files, tenant_type and document_refnumber.",
        )
    if sources and file:
        raise HTTPException(status_code=400, detail="Please send either files or file links in one request, not both.")
    if sources:
        links = [row.get("source") or "" for row in sources]
        _check_links(links)
        work = partial(
            process_detected_sources, tenant_type=tenant_type, document_refnumber=document_refnumber, sources=links,
        )
        return work, links
    if not file:
        raise HTTPException(
            status_code=400,
            detail="Please add at least one document. Attach a file, or send sources[n][source] links.",
        )
    items = [
        {
            "source": upload.filename or f"file {position}",
            "content": await upload.read(),
            "filename": upload.filename or f"document_{position}",
            "content_type": upload.content_type or "",
        }
        for position, upload in enumerate(file, start=1)
    ]
    work = partial(process_detected, tenant_type=tenant_type, document_refnumber=document_refnumber, items=items)
    return work, [item["source"] for item in items]


@router.post("", response_model=DetectOcrResponse)
async def detect_ocr_leasing(request: DetectOcrRequest) -> dict[str, Any]:
    """Find, check and read every document in the linked files."""
    work, _ = _sources_work(request)
    return await work()


@router.post("/upload", response_model=DetectOcrResponse)
async def detect_ocr_leasing_upload(
    request: Request,
    tenant_type: Annotated[str, Form(...)],
    document_refnumber: Annotated[str, Form(...)],
    file: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, Any]:
    """Find, check and read every document in the attached files (repeat `file` once per file), or
    in `sources[n][source]` links. A file may hold several documents."""
    work, _ = await _upload_work(request, tenant_type, document_refnumber, file)
    return await work()


def _start_job(work: Work, *, tenant_type: str, document_refnumber: str, sources: list[str]) -> dict[str, str]:
    """The mandatory-document rule can only be judged once the types are found, so unlike the normal
    jobs it is checked inside the job: a missing National ID / trade licence ends the job with that
    message on every document of its result."""
    job_id = jobs.start(
        work,
        partial(
            failed_detect_response,
            tenant_type=tenant_type, document_refnumber=document_refnumber, sources=sources,
            started_at=time.perf_counter(),
        ),
    )
    return {"job_id": job_id}


@router.post("/jobs", status_code=202)
async def start_detect_job(request: DetectOcrRequest) -> dict[str, str]:
    """Same body as `POST /ocr/leasing/detect`, in the background. Poll `GET /ocr/leasing/jobs/{job_id}`."""
    work, sources = _sources_work(request)
    return _start_job(work, tenant_type=request.tenant_type, document_refnumber=request.document_refnumber, sources=sources)


@router.post("/jobs/upload", status_code=202)
async def start_detect_upload_job(
    request: Request,
    tenant_type: Annotated[str, Form(...)],
    document_refnumber: Annotated[str, Form(...)],
    file: Annotated[list[UploadFile] | None, File()] = None,
) -> dict[str, str]:
    """Same form-data as `POST /ocr/leasing/detect/upload`, in the background. Poll
    `GET /ocr/leasing/jobs/{job_id}`."""
    work, sources = await _upload_work(request, tenant_type, document_refnumber, file)
    return _start_job(work, tenant_type=tenant_type, document_refnumber=document_refnumber, sources=sources)
