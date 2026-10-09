"""POST /ocr/leasing and /ocr/leasing/upload - the same pipeline as /ocr, leasing response shape.

  - `POST /ocr/leasing`         JSON body, documents named by URL - the API downloads each one.
  - `POST /ocr/leasing/upload`  multipart form-data, documents attached directly as files - or
                                named by URL in `sources[n][field]` rows, the same documents the
                                JSON route takes, for clients that can only send form-data.
  - `GET  /ocr/leasing/files?url=...`  view one uploaded `<name>_ocr.<ext>` result from blob storage
                                       (also `/ocr/leasing/files/{blob path}`).

Both POST routes return `LeasingOcrResponse`. See app.services.leasing for how a batch is judged.
"""

import time
from functools import partial
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import ValidationError

from app.api.forms import Work, form_sources
from app.schemas.leasing import LeasingOcrRequest, LeasingOcrResponse
from app.services import jobs
from app.services.leasing.batch import process_batch, process_sources
from app.services.leasing.outputs import fetch_ocr_file
from app.services.leasing.reports import failed_response
from app.services.leasing.rules import (
    ensure_documents_complete,
    ensure_unique_document_names,
    require_mandatory_document,
)

router = APIRouter(prefix="/ocr/leasing", tags=["leasing"])


@router.post("", response_model=LeasingOcrResponse)
async def extract_ocr_leasing(request: LeasingOcrRequest) -> dict[str, Any]:
    """Run the OCR pipeline over documents given as downloadable URLs."""
    return await process_sources(request)




async def _upload_work(
    request: Request,
    tenant_type: str,
    document_refnumber: str,
    file: list[UploadFile] | None,
    document_id: list[str] | None,
    document_type: list[str] | None,
    document_name: list[str] | None,
) -> Work:
    """Checks a form-data request and reads its files, returning the processing still to run.

    The files are read here, while the request is still open, so the returned work can also run in
    the background after the response has been sent.
    """
    sources = form_sources(await request.form())
    file, document_id = file or [], document_id or []
    document_type, document_name = document_type or [], document_name or []

    if sources:
        if file:
            raise HTTPException(
                status_code=400,
                detail="Please send either files or file links in one request, not both.",
            )
        try:
            source_request = LeasingOcrRequest(
                tenant_type=tenant_type, document_refnumber=document_refnumber, sources=sources,
            )
        except ValidationError as exc:
            # Same 422 shape as any other malformed body, with the form's own key names in loc.
            raise RequestValidationError(
                [{"type": err["type"], "msg": err["msg"], "loc": ("body", f"sources[{err['loc'][1]}][{err['loc'][2]}]")}
                 if len(err["loc"]) == 3 else {"type": err["type"], "msg": err["msg"], "loc": ("body", *err["loc"])}
                 for err in exc.errors()]
            ) from exc
        return partial(process_sources, source_request)

    if not file:
        raise HTTPException(
            status_code=400,
            detail="Please add at least one document. Attach a file (with its document_id, document_type and document_name), "
                   "or send sources[n][source] URLs",
        )
    if not len(file) == len(document_id) == len(document_type) == len(document_name):
        raise HTTPException(
            status_code=400,
            detail="Every file needs its own document_id, document_type and document_name. Please check that none are missing.",
        )

    items: list[dict[str, Any]] = []
    for upload, doc_id, doc_type, doc_name in zip(file, document_id, document_type, document_name):
        content = await upload.read()
        items.append({
            "source": upload.filename or doc_name,
            "content": content,
            "filename": upload.filename or doc_name,
            "content_type": upload.content_type or "",
            "document_id": doc_id,
            "document_type": doc_type,
            "document_name": doc_name,
        })

    return partial(process_batch, tenant_type=tenant_type, document_refnumber=document_refnumber, items=items)


@router.post("/upload", response_model=LeasingOcrResponse)
async def extract_ocr_leasing_upload(
    request: Request,
    tenant_type: Annotated[str, Form(...)],
    document_refnumber: Annotated[str, Form(...)],
    file: Annotated[list[UploadFile] | None, File()] = None,
    document_id: Annotated[list[str] | None, Form()] = None,
    document_type: Annotated[list[str] | None, Form()] = None,
    document_name: Annotated[list[str] | None, Form()] = None,
) -> dict[str, Any]:
    """Run the OCR pipeline over documents sent as form-data - attached files, or URLs.

    **Files:** send one `file`, `document_id`, `document_type` and `document_name` field per
    document, all using those same repeated key names - they are paired up by position, first
    `file` with first `document_id` and so on.

    **URLs:** send `sources[0][source]`, `sources[0][document_id]`, `sources[0][document_type]`,
    `sources[0][document_name]`, then `sources[1][...]` and so on - exactly what the JSON route
    takes, downloaded the same way.

    `tenant_type` and `document_refnumber` are sent once per request. One request uses one style.
    """
    work = await _upload_work(
        request, tenant_type, document_refnumber, file, document_id, document_type, document_name,
    )
    return await work()


# --- Background jobs, for a progress bar ---------------------------------------------------------
#
#   POST /ocr/leasing/jobs          same JSON body as POST /ocr/leasing        -> {"job_id": ...}
#   POST /ocr/leasing/jobs/upload   same form-data as POST /ocr/leasing/upload -> {"job_id": ...}
#   GET  /ocr/leasing/jobs/{job_id} -> {"jobId", "percentage", "status", "result"}
#
# Poll the GET until status is Completed: result is null while Processing, then the same response
# the normal endpoint gives. Percentage reaches 100 when the job has ended, whether it succeeded or
# failed; the outcome and any errors are in result.


def _check_before_starting(tenant_type: str, documents: list[dict[str, Any]], *, check_source: bool) -> None:
    """The cheap request checks the processing would make first - run before a job is created, so a
    request that can never succeed is rejected straight away instead of as a finished job."""
    ensure_documents_complete(documents, check_source=check_source)
    ensure_unique_document_names([document.get("document_name") or "" for document in documents])
    require_mandatory_document(tenant_type, [document.get("document_type") or "" for document in documents])


def _start_job(work: Work, *, tenant_type: str, document_refnumber: str, documents: list[dict[str, Any]]) -> dict[str, str]:
    started_at = time.perf_counter()
    job_id = jobs.start(
        work,
        partial(
            failed_response,
            tenant_type=tenant_type, document_refnumber=document_refnumber, documents=documents,
            started_at=started_at,
        ),
    )
    return {"job_id": job_id}


@router.post("/jobs", status_code=202)
async def start_ocr_leasing_job(request: LeasingOcrRequest) -> dict[str, str]:
    """Start processing documents given as URLs (same body as `POST /ocr/leasing`) in the background."""
    documents = [source.model_dump() for source in request.sources]
    _check_before_starting(request.tenant_type, documents, check_source=True)
    return _start_job(
        partial(process_sources, request),
        tenant_type=request.tenant_type, document_refnumber=request.document_refnumber, documents=documents,
    )


@router.post("/jobs/upload", status_code=202)
async def start_ocr_leasing_upload_job(
    request: Request,
    tenant_type: Annotated[str, Form(...)],
    document_refnumber: Annotated[str, Form(...)],
    file: Annotated[list[UploadFile] | None, File()] = None,
    document_id: Annotated[list[str] | None, Form()] = None,
    document_type: Annotated[list[str] | None, Form()] = None,
    document_name: Annotated[list[str] | None, Form()] = None,
) -> dict[str, str]:
    """Start processing documents sent as form-data (same fields as `POST /ocr/leasing/upload`) in
    the background."""
    work = await _upload_work(
        request, tenant_type, document_refnumber, file, document_id, document_type, document_name,
    )
    sources = form_sources(await request.form())
    if sources:
        documents = sources
        _check_before_starting(tenant_type, documents, check_source=True)
    else:
        documents = [
            {"source": upload.filename or doc_name, "document_id": doc_id,
             "document_type": doc_type, "document_name": doc_name}
            for upload, doc_id, doc_type, doc_name in zip(
                file or [], document_id or [], document_type or [], document_name or [],
            )
        ]
        _check_before_starting(tenant_type, documents, check_source=False)
    return _start_job(work, tenant_type=tenant_type, document_refnumber=document_refnumber, documents=documents)


@router.get("/jobs/{job_id}")
async def get_ocr_leasing_job(job_id: str) -> dict[str, Any]:
    """Progress and result in one call: `{"jobId", "percentage", "status", "result"}`.

    Processing (percentage 0-99, result null) while the documents are being checked; Completed
    (percentage 100) once finished, with result holding the same response the normal endpoint
    gives - success or failed, with any errors on their documents.
    """
    return jobs.get_status(job_id)


_FILE_RESPONSES = {200: {"content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}, "application/pdf": {}}}}


def _inline(content: bytes, content_type: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"},
    )


@router.get("/files", response_class=Response, responses=_FILE_RESPONSES)
async def view_ocr_file_by_url(url: str) -> Response:
    """Show one uploaded `<name>_ocr.<ext>` result - pass a source's `ocr_file` URL as `url`.

    Served inline, so a browser or Postman displays the image or PDF directly.
    """
    return _inline(*await fetch_ocr_file(url))


@router.get(
    "/files/{blob_path:path}",
    response_class=Response,
    responses=_FILE_RESPONSES,
)
async def view_ocr_file(blob_path: str) -> Response:
    """Show one uploaded `<name>_ocr.<ext>` result by its blob path inside the container."""
    return _inline(*await fetch_ocr_file(blob_path))
