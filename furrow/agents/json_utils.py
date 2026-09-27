"""Utilities for extracting JSON from LLM text responses.

Language models frequently wrap JSON output in markdown code fences or
prepend/append explanatory prose.  The helpers here normalise such responses
so callers can safely call ``json.loads``.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json|python|yaml|text)?\s*(.+?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Return the substring of *text* that should be valid JSON.

    Strategy:
    1. If the text is wrapped in markdown code fences, extract the fenced
       content.
    2. Otherwise, look for the outermost balanced ``{...}`` or ``[...]`` block.
    3. Fall back to the original text stripped of surrounding whitespace.
    """
    text = text.strip()

    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # Try object syntax.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]

    # Try array syntax.
    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]

    return text


def parse_json_response(text: str, *, default=None):
    """Parse *text* as JSON, attempting extraction if raw parsing fails.

    Returns *default* when the text cannot be parsed.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    extracted = extract_json(text)
    try:
        return json.loads(extracted)
    except (json.JSONDecodeError, ValueError):
        return default
