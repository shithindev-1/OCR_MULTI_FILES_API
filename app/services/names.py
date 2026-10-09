"""Tolerant comparison of English names read off different documents."""

import re

_NAME_TOKEN = re.compile(r"[A-Za-z]+")


def name_tokens(name: object) -> set[str]:
    """The name's parts, upper-cased, ignoring single letters and anything not a letter."""
    return {token.upper() for token in _NAME_TOKEN.findall(str(name or "")) if len(token) > 1}


def same_name(first: object, second: object) -> bool:
    """Every part of the shorter name appears in the longer one, with at least two parts in common -
    case, word order and middle names ignored."""
    a, b = name_tokens(first), name_tokens(second)
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 2 and shorter <= longer
