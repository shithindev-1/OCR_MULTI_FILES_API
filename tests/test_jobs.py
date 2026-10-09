"""Background jobs and their progress percentage - Gemini is stubbed out throughout."""

import asyncio
import io
import time
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.core.config import settings
from app.main import app
from app.schemas.ocr import DATA_FIELDS
from app.services import blob_storage, jobs
from app.services.gemini import get_gemini_service
from app.services.progress import ProgressTracker

JOBS_URL = "/api/v1/ocr/leasing/jobs"


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _raw(types_present: list[str], **fields: Any) -> dict[str, Any]:
    data = dict.fromkeys(DATA_FIELDS)
    data.update(fields)
    return {"data": data, "documentComplete": True, "documentTypesPresent": types_present,
            "frontVisible": True, "backVisible": True}


# --- the percentage itself -------------------------------------------------------------------------

def test_progress_fills_per_document_and_only_reaches_100_when_finished() -> None:
    tracker = ProgressTracker()
    tracker.report("download", 1, 1)
    assert tracker.percent == 10
    tracker.report("extraction", 1, 4)  # one of four documents read
    assert tracker.percent == 17
    tracker.report("extraction", 4, 4)
    assert tracker.percent == 40
    tracker.report("crop_rotate_upload", 1, 1)  # every stage done...
    assert tracker.percent == 99  # ...but not 100 until the job has ended
    tracker.finish()
    assert tracker.percent == 100


def test_progress_never_moves_backwards() -> None:
    tracker = ProgressTracker()
    tracker.report("self_verification", 1, 1)
    tracker.report("extraction", 1, 4)
    assert tracker.percent == 65


# --- the job store ---------------------------------------------------------------------------------

def _failed(message: str) -> dict[str, Any]:
    return {"status": "failed", "documents": [{"error": message}]}


def test_a_job_reports_processing_then_completed_with_its_result() -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def work() -> dict[str, Any]:
            await release.wait()
            return {"status": "success"}

        job_id = jobs.start(work, _failed)
        await asyncio.sleep(0)
        running = jobs.get_status(job_id)
        assert running["jobId"] == job_id and running["percentage"] < 100
        assert running["status"] == "Processing" and running["result"] is None

        release.set()
        await jobs._jobs[job_id].task
        assert jobs.get_status(job_id) == {
            "jobId": job_id, "percentage": 100, "status": "Completed", "result": {"status": "success"},
        }

    asyncio.run(scenario())


def test_a_rejected_job_still_gives_the_normal_response_shape_with_its_message() -> None:
    async def scenario() -> None:
        async def work() -> dict[str, Any]:
            raise HTTPException(status_code=400, detail="A trade licence is required for a company.")

        job_id = jobs.start(work, _failed)
        await jobs._jobs[job_id].task
        status = jobs.get_status(job_id)
        assert status["status"] == "Completed" and status["percentage"] == 100
        # Never a bare {"detail": ...}: the usual response, failed, with the reason on the document.
        assert status["result"] == _failed("A trade licence is required for a company.")

    asyncio.run(scenario())


def test_a_job_that_breaks_on_our_side_gives_a_general_friendly_message() -> None:
    async def scenario() -> None:
        async def work() -> dict[str, Any]:
            raise RuntimeError("provider exploded")

        job_id = jobs.start(work, _failed)
        await jobs._jobs[job_id].task
        assert jobs.get_status(job_id)["result"] == _failed(jobs.COULD_NOT_FINISH)

    asyncio.run(scenario())


def test_an_unknown_job_is_a_404() -> None:
    with pytest.raises(HTTPException) as exc:
        jobs.get_status("no-such-job")
    assert exc.value.status_code == 404


# --- the HTTP endpoints ----------------------------------------------------------------------------

@pytest.fixture
def live_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """A client whose event loop stays alive between requests, so a background job keeps running."""
    async def no_warm_up() -> None:
        return None

    monkeypatch.setattr(main_module, "_warm_up_connections", no_warm_up)
    monkeypatch.setattr(settings, "output_retention_hours", 0)
    monkeypatch.setattr(settings, "require_https", False)
    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    monkeypatch.setattr(blob_storage, "get_blob_storage", lambda: None)
    with TestClient(app) as client:
        yield client


def _passport_and_national_id() -> dict[str, Any]:
    """Form-data for an individual job: a passport plus the National ID every individual request needs."""
    return {
        "files": [("file", ("passport.png", _png(), "image/png")), ("file", ("nid.png", _png(), "image/png"))],
        "data": {
            "tenant_type": "individual", "document_refnumber": "job-1",
            "document_id": ["1", "2"], "document_type": ["passport", "national_id"],
            "document_name": ["Passport", "National ID"],
        },
    }


def _wait_for_100(client: TestClient, job_id: str) -> list[int]:
    seen: list[int] = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        value = client.get(f"{JOBS_URL}/{job_id}").json()["percentage"]
        seen.append(value)
        if value == 100:
            return seen
        time.sleep(0.05)
    raise AssertionError(f"job never reached 100: {seen}")


def test_upload_job_returns_progress_then_the_normal_response_in_one_endpoint(
    live_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        if kwargs["filename"] == "nid.png":
            return _raw(["national_id"])
        return _raw(["passport"], Passport_Number="P1234567")

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)

    started = live_client.post(f"{JOBS_URL}/upload", **_passport_and_national_id())
    assert started.status_code == 202
    job_id = started.json()["job_id"]

    first = live_client.get(f"{JOBS_URL}/{job_id}").json()
    assert sorted(first) == ["jobId", "percentage", "result", "status"]
    assert _wait_for_100(live_client, job_id)[-1] == 100

    final = live_client.get(f"{JOBS_URL}/{job_id}")
    assert final.status_code == 200
    job = final.json()
    assert job["status"] == "Completed" and job["percentage"] == 100
    assert job["result"]["status"] == "success"
    assert job["result"]["form_data"]["passport_number"] == "P1234567"
    # The separate /progress endpoint is gone - progress and result come from this one call.
    assert live_client.get(f"{JOBS_URL}/{job_id}/progress").status_code == 404


def test_a_failed_job_reaches_100_and_the_error_is_in_its_result(
    live_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw(["passport"])  # sent as a National ID

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)

    job_id = live_client.post(f"{JOBS_URL}/upload", files=[("file", ("id.png", _png(), "image/png"))], data={
        "tenant_type": "individual", "document_refnumber": "job-2",
        "document_id": "1", "document_type": "national_id", "document_name": "National ID",
    }).json()["job_id"]

    assert _wait_for_100(live_client, job_id)[-1] == 100
    job = live_client.get(f"{JOBS_URL}/{job_id}").json()
    assert job["status"] == "Completed"
    assert job["result"]["status"] == "failed"
    assert "does not look like a National ID" in job["result"]["documents"][0]["error"]


def test_a_company_job_without_a_trade_licence_is_rejected_before_it_starts(live_client: TestClient) -> None:
    response = live_client.post(f"{JOBS_URL}/upload", files=[("file", ("p.png", _png(), "image/png"))], data={
        "tenant_type": "company", "document_refnumber": "job-3",
        "document_id": "1", "document_type": "passport", "document_name": "Passport",
    })
    assert response.status_code == 400
    assert "trade licence is required" in response.json()["detail"]


def test_an_individual_job_without_a_national_id_is_rejected_before_it_starts(live_client: TestClient) -> None:
    response = live_client.post(f"{JOBS_URL}/upload", files=[("file", ("p.png", _png(), "image/png"))], data={
        "tenant_type": "individual", "document_refnumber": "job-6",
        "document_id": "1", "document_type": "passport", "document_name": "Passport",
    })
    assert response.status_code == 400
    assert response.json()["detail"] == (
        "A National ID is required for an individual. Please upload the tenant's National ID and try again."
    )
    assert "job_id" not in response.json()


def test_a_json_job_with_an_empty_source_is_rejected_before_it_starts(live_client: TestClient) -> None:
    response = live_client.post(JOBS_URL, json={
        "tenant_type": "individual", "document_refnumber": "job-5",
        "sources": [{"source": "", "document_id": "1", "document_type": "passport", "document_name": "pp"}],
    })
    assert response.status_code == 422
    assert response.json() == {
        "detail": "The file link for 'pp' is empty. Please add the document's link or file path and try again."
    }
    assert "job_id" not in response.json()


def test_polling_progress_never_counts_against_the_rate_limit(
    live_client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return _raw(["national_id" if kwargs["filename"] == "nid.png" else "passport"])

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)
    monkeypatch.setattr(settings, "rate_limit_requests", 1)

    job_id = live_client.post(f"{JOBS_URL}/upload", **_passport_and_national_id()).json()["job_id"]

    for _ in range(5):
        assert live_client.get(f"{JOBS_URL}/{job_id}").status_code == 200
