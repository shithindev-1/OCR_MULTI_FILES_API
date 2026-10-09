"""In-memory reuse of Gemini results for byte-identical files.

A file is matched by its SHA-256, so a hit is only ever the exact same bytes sent again - a retry,
a re-run test, the same passport attached to another application. Entries live in process memory
only (never on disk), expire after `settings.result_cache_seconds`, and vanish on restart; the
oldest are evicted first once `_MAX_ENTRIES` is reached, so memory stays bounded.

Values are deep-copied in and out, so a caller mutating what it got back can never change what the
next request receives.
"""

import copy
import logging
import time
from collections import OrderedDict
from typing import Any

from app.core.config import settings

logger = logging.getLogger("uae_ocr")

_MAX_ENTRIES = 500

_entries: "OrderedDict[tuple[Any, ...], tuple[float, Any]]" = OrderedDict()


def get(key: tuple[Any, ...]) -> Any | None:
    """The cached value for `key`, or None when absent, expired, or caching is off."""
    if settings.result_cache_seconds <= 0:
        return None
    entry = _entries.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.monotonic() >= expires_at:
        del _entries[key]
        return None
    return copy.deepcopy(value)


def put(key: tuple[Any, ...], value: Any) -> None:
    if settings.result_cache_seconds <= 0:
        return
    _entries[key] = (time.monotonic() + settings.result_cache_seconds, copy.deepcopy(value))
    _entries.move_to_end(key)
    while len(_entries) > _MAX_ENTRIES:
        _entries.popitem(last=False)


def clear() -> None:
    _entries.clear()
