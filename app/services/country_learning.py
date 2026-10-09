"""New country names, learned at runtime.

A country value the curated lists in app/data do not know is returned as extracted, and saved to a
new-names list - a 2- or 3-letter value to new_country_codes.json, anything else to
new_nationalities.json - in settings.country_learning_dir. In the background, Gemini gives the
value's 2-letter country code and app/data/country_codes.json turns that code into the exact
country name, so the response is never delayed and the prompt never carries the country list; the
next request carrying the same value is converted from the saved entry, with no model call.

Each file maps the value as it was read to its entry:

    {"Syrian Arab Rep": {"name": "Syria", "code": "SY", "source": "model", "added": "2026-10-08"}}

`name` is null when the value is not a country, or its code is not in the client's list (`code`
then shows what the model answered); the value is never asked about again. Entries can be corrected or filled in by hand (a hand edit takes effect without a
restart), and good ones moved into app/data for good.
"""

import asyncio
import json
import logging
import os
import threading
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, settings
from app.services.countries import EXACT_NAMES, is_code, lookup_key, name_for_code
from app.services.gemini import get_gemini_service

logger = logging.getLogger("uae_ocr")

CODES_FILE = "new_country_codes.json"
NATIONALITIES_FILE = "new_nationalities.json"


def _directory() -> Path:
    path = Path(settings.country_learning_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


class _NewNamesList:
    """One new-names file, re-read whenever it changes on disk."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self._entries: dict[str, Any] = {}
        self._names: dict[str, str | None] = {}
        self._loaded: tuple[Path, int | None] | None = None
        self._readable = True
        self._write_lock = threading.Lock()

    @property
    def path(self) -> Path:
        return _directory() / self.filename

    def _refresh(self) -> None:
        path = self.path
        try:
            mtime: int | None = path.stat().st_mtime_ns
        except FileNotFoundError:
            mtime = None
        if self._loaded == (path, mtime):
            return
        self._loaded = (path, mtime)
        entries: dict[str, Any] = {}
        self._readable = True
        if mtime is not None:
            try:
                entries = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(entries, dict):
                    raise ValueError("expected a JSON object")
            except (OSError, ValueError) as exc:
                # Never overwrite a file someone broke by hand: stop using and writing it until fixed.
                logger.error("Ignoring %s until it is fixed: %s", self.path, exc)
                entries, self._readable = {}, False
        names: dict[str, str | None] = {}
        for value, entry in entries.items():
            name = entry.get("name") if isinstance(entry, dict) else None
            if name is not None and name not in EXACT_NAMES:
                logger.warning("%s: '%s' points to '%s', which is not an exact country name", self.filename, value, name)
                name = None
            names[lookup_key(value)] = name
        # Swapped in whole, so a request reading at the same moment never sees a half-built list.
        self._entries, self._names = entries, names

    def name(self, value: str) -> str | None:
        self._refresh()
        return self._names.get(lookup_key(value))

    def knows(self, value: str) -> bool:
        self._refresh()
        return lookup_key(value) in self._names

    def add(self, value: str, name: str | None, code: str | None) -> None:
        """Blocking file write - call it from a worker thread."""
        with self._write_lock:
            self._refresh()
            if not self._readable or lookup_key(value) in self._names:
                return
            self._entries[value] = {"name": name, "code": code, "source": "model", "added": date.today().isoformat()}
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside the real file and swapped in, so a crash mid-write never leaves half a file.
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)
            self._loaded = None
            self._refresh()


_CODES = _NewNamesList(CODES_FILE)
_NATIONALITIES = _NewNamesList(NATIONALITIES_FILE)

_in_progress: set[str] = set()
# Strong references, so a background task is not garbage-collected before it finishes.
_tasks: set[asyncio.Task] = set()


def _list_for(value: str) -> _NewNamesList:
    return _CODES if is_code(value) else _NATIONALITIES


def learned_country_name(value: str) -> str | None:
    """The exact country name saved for a value earlier, or None."""
    if not settings.country_learning_enabled:
        return None
    return _list_for(value).name(value)


def learn_in_background(values: Iterable[str]) -> None:
    """Saves each value not seen before to its new-names list and has Gemini match it, without
    waiting. Does nothing outside a running event loop or when learning is switched off."""
    if not settings.country_learning_enabled:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    for value in values:
        value = value.strip()
        key = lookup_key(value)
        if not key or key in _in_progress or _list_for(value).knows(value):
            continue
        _in_progress.add(key)
        task = asyncio.create_task(_learn(value, key))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)


async def _learn(value: str, key: str) -> None:
    kind = "country code" if is_code(value) else "nationality or country name"
    try:
        try:
            code = await get_gemini_service().country_code(value=value, kind=kind)
        except Exception:
            # Not saved, so the value is tried again the next time it is seen.
            logger.warning("Could not match new country value %r", value, exc_info=True)
            return
        name = name_for_code(code) if code else None
        target = _list_for(value)
        await asyncio.to_thread(target.add, value, name, code)
        logger.info("New country value %r saved to %s as %s (code %s)", value, target.filename, name, code)
    finally:
        _in_progress.discard(key)
