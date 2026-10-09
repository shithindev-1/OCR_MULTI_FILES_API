"""Extracted-data logging, source-URL allowlisting, and stored-file retention."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.core.config import settings
from app.schemas.leasing import SourceDocument
from app.services.leasing import sources
from app.services.retention import purge_expired_files


def _source(url: str) -> SourceDocument:
    return SourceDocument(source=url, document_id="1", document_type="Passport", document_name="Passport")


def _download(handler: Any, url: str) -> tuple[bytes, str, str]:
    async def run() -> tuple[bytes, str, str]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await sources.download_source(client, _source(url))

    return asyncio.run(run())


def test_extracted_results_are_not_logged_by_default() -> None:
    assert type(settings).model_fields["log_extracted_results"].default is False


def test_source_url_on_an_unlisted_host_is_never_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allowed_source_hosts", "")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, content=b"x")

    with pytest.raises(sources.HTTPException) as exc:
        _download(handler, "http://169.254.169.254/metadata/a.jpg")

    assert exc.value.status_code == 400
    assert "cannot be downloaded from this location" in exc.value.detail
    assert requested == []


def test_source_url_on_a_listed_host_is_fetched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allowed_source_hosts", "files.example.com")

    content, filename, _ = _download(lambda request: httpx.Response(200, content=b"jpeg-bytes"), "https://FILES.example.com/a/Passport.jpg")

    assert content == b"jpeg-bytes"
    assert filename == "Passport.jpg"


def test_redirects_are_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allowed_source_hosts", "files.example.com")
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.host)
        return httpx.Response(302, headers={"Location": "http://10.0.0.5/secret.jpg"})

    with pytest.raises(sources.HTTPException) as exc:
        _download(handler, "https://files.example.com/a.jpg")

    assert exc.value.status_code == 502
    assert requested == ["files.example.com"]


def test_oversized_download_is_cut_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "allowed_source_hosts", "files.example.com")
    monkeypatch.setattr(settings, "max_file_size_mb", 1)

    with pytest.raises(sources.HTTPException) as exc:
        _download(lambda request: httpx.Response(200, content=b"x" * (1024 * 1024 + 1)), "https://files.example.com/a.jpg")

    assert exc.value.status_code == 413


def _file(path: Path, age_seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


def test_retention_deletes_only_expired_files_and_empty_folders(tmp_path: Path) -> None:
    old = _file(tmp_path / "out" / "ref-1" / "16_ID" / "original_id.jpg", age_seconds=7200)
    fresh = _file(tmp_path / "out" / "ref-2" / "17_ID" / "original_id.jpg", age_seconds=60)
    crop = _file(tmp_path / "crops" / "id_abc.png", age_seconds=7200)

    deleted = purge_expired_files([tmp_path / "out", tmp_path / "crops", tmp_path / "missing"], max_age_seconds=3600)

    assert deleted == 2
    assert not old.exists() and not crop.exists()
    assert fresh.exists()
    assert not (tmp_path / "out" / "ref-1").exists()
    assert (tmp_path / "out").is_dir() and (tmp_path / "crops").is_dir()
