"""Tolerant comparison of English names read off different documents."""

import re

_NAME_TOKEN = re.compile(r"[A-Za-z]+")
# A name part printed with or without a space ("ABD ALRAZAK" / "ABDALRAZAK") is one part: up to this many
# neighbouring words may join into one.
_MAX_JOINED_WORDS = 3


def _ordered_tokens(name: object) -> list[str]:
    return [token.upper() for token in _NAME_TOKEN.findall(str(name or "")) if len(token) > 1]


def name_tokens(name: object) -> set[str]:
    """The name's parts, upper-cased, ignoring single letters and anything not a letter."""
    return set(_ordered_tokens(name))


def _joined_parts(tokens: list[str]) -> set[str]:
    """Every part of the name, and every run of up to _MAX_JOINED_WORDS neighbouring parts written together."""
    return {
        "".join(tokens[start:start + size])
        for start in range(len(tokens))
        for size in range(1, _MAX_JOINED_WORDS + 1)
        if start + size <= len(tokens)
    }


def _parts_found(name: list[str], other: list[str]) -> int:
    """How many parts `name` has when each part (or a few neighbouring words written together) is
    found in `other` - spaces ignored either way. 0 when any part is missing."""
    available = _joined_parts(other)
    found, index = 0, 0
    while index < len(name):
        size = next(
            (size for size in range(1, _MAX_JOINED_WORDS + 1)
             if index + size <= len(name) and "".join(name[index:index + size]) in available),
            0,
        )
        if not size:
            return 0
        found += 1
        index += size
    return found


def same_name(first: object, second: object, *, ignore_spacing: bool = True) -> bool:
    """Every part of the shorter name appears in the longer one, with at least two parts in common -
    case, word order and middle names ignored. With `ignore_spacing`, a part printed with or without a
    space counts as the same part ("ABD ALRAZAK MAMOUN" = "Abdalrazak Mamoun")."""
    a, b = name_tokens(first), name_tokens(second)
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 2 and shorter <= longer:
        return True
    if not ignore_spacing:
        return False
    x, y = _ordered_tokens(first), _ordered_tokens(second)
    return _parts_found(x, y) >= 2 or _parts_found(y, x) >= 2
