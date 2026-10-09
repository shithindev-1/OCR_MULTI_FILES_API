"""Deletes stored document files once they pass OUTPUT_RETENTION_HOURS.

Covers everything this service writes derived from an upload: `detection_output_dir` (crops,
metadata) and `leasing_output_dir` (originals, crops, merged PDFs). Logs have their own daily
rotation and `LOG_RETENTION_DAYS`, so they are not touched here.
"""

import asyncio
import logging
import time
from pathlib import Path

from app.core.config import Settings

logger = logging.getLogger("uae_ocr")

SWEEP_INTERVAL_SECONDS = 3600


def purge_expired_files(directories: list[Path], max_age_seconds: float, now: float | None = None) -> int:
    """Deletes every file under `directories` last modified more than `max_age_seconds` ago, then
    any folder left empty. Returns the number of files deleted."""
    cutoff = (now if now is not None else time.time()) - max_age_seconds
    deleted = 0
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError as exc:
                # A file still open elsewhere (Windows) is retried on the next sweep.
                logger.warning("Retention sweep could not delete a file error_type=%s", type(exc).__name__)
        # Deepest first, so a parent emptied by its children's removal goes too. The root stays.
        for folder in sorted((p for p in directory.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                folder.rmdir()
            except OSError:
                pass  # not empty
    return deleted


def output_directories(settings: Settings) -> list[Path]:
    return [Path(settings.detection_output_dir), Path(settings.leasing_output_dir)]


async def run_retention_sweeps(settings: Settings) -> None:
    """Sweeps at startup and then every SWEEP_INTERVAL_SECONDS until cancelled."""
    max_age = settings.output_retention_hours * 3600
    while True:
        try:
            deleted = await asyncio.to_thread(purge_expired_files, output_directories(settings), max_age)
            if deleted:
                logger.info("Retention sweep deleted files=%d older_than_hours=%d", deleted, settings.output_retention_hours)
        except Exception:
            logger.exception("Retention sweep failed")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
