"""Shared test setup: keep a developer's own .env from changing what the tests exercise."""

import os
import tempfile

# Before the app is imported: test runs write their logs to a throwaway folder, never into the
# real logs/ the service and its developers read.
os.environ["LOG_DIR"] = tempfile.mkdtemp(prefix="ocr-test-logs-")
# Required settings, so the tests also run in a fresh clone with no .env. Gemini is always stubbed.
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("GEMINI_MODEL", "test-model")

import pytest  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.services import ocr_pipeline, result_cache  # noqa: E402
from app.services.validation import _rate_events  # noqa: E402


@pytest.fixture(autouse=True)
def _plain_http_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # The test client talks plain HTTP; tests about HTTPS switch it back on themselves.
    monkeypatch.setattr(settings, "require_https", False)


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real API_KEYS in .env would otherwise reject every test request with 401.
    monkeypatch.setattr(settings, "api_keys", "")


@pytest.fixture(autouse=True)
def _fresh_rate_limiter() -> None:
    # The limiter is process-wide; without a reset, a long test run trips its 30-per-minute cap.
    _rate_events.clear()


@pytest.fixture(autouse=True)
def _fresh_result_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most tests upload the same generated image bytes under different names and stub a different
    # answer for each, which the cache would rightly treat as one file. Off by default here; the
    # cache's own tests switch it on.
    result_cache.clear()
    monkeypatch.setattr(settings, "result_cache_seconds", 0)
    # The one retry after a failed verification call waits a second in production; not here.
    monkeypatch.setattr(ocr_pipeline, "_VERIFICATION_RETRY_DELAY_SECONDS", 0)


@pytest.fixture(autouse=True)
def _country_learning_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # Learning calls Gemini and writes new-names files; off unless a test switches it on itself.
    monkeypatch.setattr(settings, "country_learning_enabled", False)
