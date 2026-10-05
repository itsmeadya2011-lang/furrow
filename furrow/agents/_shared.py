"""Coercion helpers shared by the planner and the tester.

Both agents read loosely-typed JSON off a model and have to hand pydantic something it will
accept, and both have to keep the text they echo back bounded. Those two rules are identical
in both modules, so they live here once; the module itself is the private boundary.
"""

from __future__ import annotations

from typing import Any


def truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}... [truncated, {len(text)} chars]"


def as_text(value: Any) -> str:
    """Render a JSON scalar as plain text; containers and ``None`` yield ``""``."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def as_str_list(value: Any) -> list[str]:
    """Coerce a list-ish model field into a list of non-empty strings.

    A bare string is the single-element case models sometimes emit, and unusable entries are
    dropped rather than passed on to pydantic as the wrong type.
    """
    if value is None or isinstance(value, (dict, bool)):
        return []
    items = list(value) if isinstance(value, (list, tuple, set)) else [value]
    return [text for text in (as_text(item) for item in items) if text]
