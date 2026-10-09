"""Form-data helpers shared by the leasing routes."""

import re
from collections.abc import Awaitable, Callable
from typing import Any

# The processing a route has prepared, still to run - now, or later as a background job.
Work = Callable[[], Awaitable[dict[str, Any]]]

# `sources[0][source]`, `sources[0][document_id]`, ... - the bracket notation form builders use for
# a list of objects. `sources[0].source` is accepted too.
_SOURCE_FIELD = re.compile(r"^sources\[(\d+)\](?:\[(\w+)\]|\.(\w+))$")


def form_sources(form: Any) -> list[dict[str, str]]:
    """The `sources[n][field]` rows of a form, as a list of source dicts ordered by n."""
    by_index: dict[int, dict[str, str]] = {}
    for key, value in form.multi_items():
        match = _SOURCE_FIELD.match(key)
        if match and isinstance(value, str):
            by_index.setdefault(int(match[1]), {})[match[2] or match[3]] = value
    return [by_index[index] for index in sorted(by_index)]
