"""Tolerant JSON extraction for LLM replies.

Models rarely emit bare JSON.  A reply that should have been a JSON object is
routinely wrapped in a markdown code fence (```json ... ```), padded with a
sentence of prose on either side, or both.  ``json.loads`` rejects all of those
shapes, which used to make planning and testing fail on perfectly good replies.

``parse_json_object`` recovers the object in the widest set of shapes it can
recognise and raises a descriptive ``ValueError`` when nothing parses, so callers
never have to guess whether a failure came from the model or from the parser.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

__all__ = ["parse_json_object", "strip_code_fences"]

#: Maximum number of characters of the offending reply quoted in error messages.
MAX_EXCERPT_CHARS: Final[int] = 200

# Both the optional language tag and the body are captured: callers unpack
# ``findall`` results as ``(language, body)`` pairs.
_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\r?\n?(.*?)(?:\r?\n)?[ \t]*```",
    re.DOTALL,
)


def _excerpt(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Return a whitespace-collapsed, length-bounded excerpt of ``text``."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def strip_code_fences(text: str) -> str:
    """Return the body of the first non-empty markdown code fence in ``text``.

    Falls back to the whitespace-stripped input when the text holds no fenced
    block, so this is always safe to call before further parsing.
    """
    for _language, body in _FENCE_RE.findall(text):
        stripped = body.strip()
        if stripped:
            return stripped
    return text.strip()


def _balanced_block(text: str) -> str | None:
    """Return the outermost balanced ``{...}`` block of ``text``, if any.

    Braces that appear inside JSON string literals are ignored, which is what
    makes replies such as ``{"summary": "use {x} where needed"}`` parse.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _candidates(text: str) -> list[str]:
    """Return de-duplicated JSON-ish substrings of ``text``, most likely first."""
    bodies = [body.strip() for _language, body in _FENCE_RE.findall(text)]
    sources = [text.strip(), *bodies]
    balanced = [block for source in sources if (block := _balanced_block(source)) is not None]

    unique: list[str] = []
    for candidate in [*sources, *balanced]:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse ``text`` into a JSON object, tolerating fences and surrounding prose.

    Args:
        text: The raw model reply.

    Returns:
        The decoded mapping.

    Raises:
        ValueError: If no candidate substring of ``text`` decodes to a JSON
            object.  The message names every failure reason and quotes a
            truncated excerpt of the reply so the cause is diagnosable.
    """
    reasons: list[str] = []
    for candidate in _candidates(text):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError as exc:
            reasons.append(str(exc))
            continue
        if isinstance(decoded, dict):
            return decoded
        reasons.append(f"top-level value is {type(decoded).__name__}, expected an object")

    detail = "; ".join(reasons) if reasons else "no JSON object found"
    raise ValueError(
        f"Could not parse a JSON object from the model reply ({detail})."
        f" Reply excerpt: {_excerpt(text)!r}"
    )
