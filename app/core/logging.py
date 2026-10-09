"""Log file setup: rotating per-purpose log files with UTC/UAE/IST timestamps."""

import logging
from datetime import datetime, timedelta, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from app.core.config import PROJECT_ROOT, Settings

# Fixed offsets rather than zoneinfo: neither zone observes DST, so the offset is exact all year,
# and Windows has no IANA database for ZoneInfo to read without the extra tzdata package.
UAE_TIME = timezone(timedelta(hours=4))  # Asia/Dubai
IST_TIME = timezone(timedelta(hours=5, minutes=30))  # Asia/Kolkata


def log_timestamps() -> dict[str, str]:
    """The three stamps every JSON log line carries.

    UTC stays the canonical one — it is what sorts correctly and what correlates with anything
    outside this service — with the two local readings alongside it so a log can be read against
    a wall clock in either office without converting in your head.
    """
    now = datetime.now(timezone.utc)
    return {
        "timestamp": now.isoformat(),
        "timestamp_uae": now.astimezone(UAE_TIME).isoformat(),
        "timestamp_ist": now.astimezone(IST_TIME).isoformat(),
    }


class MultiZoneFormatter(logging.Formatter):
    """Stamps the plain-text logs with UTC, UAE, and IST on every line.

    The JSON logs carry the three as separate fields; these lines have only the one asctime slot,
    so all three are rendered into it. Each keeps its own date — past 20:00 UTC the UAE and Indian
    dates have already rolled over, and a bare time would read as the wrong day.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        moment = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return (
            f"{moment.isoformat(timespec='milliseconds')}"
            f" | UAE {moment.astimezone(UAE_TIME).strftime('%Y-%m-%d %H:%M:%S')}"
            f" | IST {moment.astimezone(IST_TIME).strftime('%Y-%m-%d %H:%M:%S')}"
        )


def _configure_file_logger(
    logger_name: str,
    log_path: Path,
    settings: Settings,
    formatter: logging.Formatter,
    *,
    level: int = logging.INFO,
    propagate: bool = True,
) -> None:
    logger = logging.getLogger(logger_name)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.propagate = propagate
    resolved_path = str(log_path.resolve())
    already_attached = any(
        isinstance(existing, TimedRotatingFileHandler) and getattr(existing, "baseFilename", None) == resolved_path
        for existing in logger.handlers
    )
    if not already_attached:
        handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            backupCount=settings.log_retention_days,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def configure_logging(settings: Settings) -> None:
    # Anchored to the project root, so a service started from another directory logs in the same place.
    log_directory = Path(settings.log_dir) if settings.log_dir else PROJECT_ROOT / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    _configure_file_logger(
        "uae_ocr",
        log_directory / "ocr-api.log",
        settings,
        MultiZoneFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )
    # Same logger, second handler: WARNING+ records are duplicated here for fast incident triage.
    _configure_file_logger(
        "uae_ocr",
        log_directory / "errors.log",
        settings,
        MultiZoneFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"),
        level=logging.WARNING,
    )
    # One JSON object per line: hit_id/timestamp/tokens/duration for every Gemini call.
    _configure_file_logger(
        "uae_ocr.hits",
        log_directory / "hits.log",
        settings,
        logging.Formatter("%(message)s"),
        propagate=False,
    )
    # One JSON object per line: hit_id + the final parsed result for that hit.
    _configure_file_logger(
        "uae_ocr.results",
        log_directory / "results.log",
        settings,
        logging.Formatter("%(message)s"),
        propagate=False,
    )
    # One JSON object per line: input/output tokens split by thinking vs. image vs. pdf for every Gemini call.
    _configure_file_logger(
        "uae_ocr.token_breakdown",
        log_directory / "token-breakdown.log",
        settings,
        logging.Formatter("%(message)s"),
        propagate=False,
    )
    # One JSON object per line: every model call one request made, each under its own key, with
    # per-pipeline and overall totals. The only file that accounts for a whole request - hits.log
    # and token-breakdown.log cover the extraction call alone.
    _configure_file_logger(
        "uae_ocr.usage",
        log_directory / "usage.log",
        settings,
        logging.Formatter("%(message)s"),
        propagate=False,
    )
    # One JSON object per line: the rotation Gemini asked for and what OpenCV actually applied,
    # per cropped document, so bad rotations can be audited without re-running anything.
    _configure_file_logger(
        "uae_ocr.rotation",
        log_directory / "rotation.log",
        settings,
        logging.Formatter("%(message)s"),
        propagate=False,
    )