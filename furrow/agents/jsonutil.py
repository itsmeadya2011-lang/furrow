"""Tolerant extraction of JSON payloads from language-model responses.

Language models routinely wrap JSON in markdown fences, prefix it with prose,
suffix it with an offer to help, or emit trailing commas. Every caller in the
agent layer funnels its ``complete()`` output through :func:`extract_json` so
that a malformed-but-recoverable reply does not throw away the whole response.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator

from furrow.agents.errors import AgentError

__all__ = ["dumps", "extract_json"]

#: Upper bound on how much text we are willing to scan, guarding against
#: pathological model output.
_MAX_SCAN = 200_000
#: How many balanced-scan start positions we try before giving up.
_MAX_STARTS = 20
#: How many markdown fences we unwrap before giving up.
_MAX_FENCES = 20
#: Characters of the offending text embedded in the error message.
_ERROR_SNIPPET = 500

_MISSING = object()
_FENCE = "```"
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def dumps(value: Any) -> str:
    """Serialise ``value`` to JSON, tolerating non-serialisable members.

    Args:
        value: Any JSON-ish value.

    Returns:
        An indented, UTF-8 friendly JSON document.
    """
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def extract_json(text: str) -> Any:
    """Recover a JSON value from arbitrary model output.

    The following strategies are attempted in order: direct parsing, markdown
    fence unwrapping (including fences nested inside prose and unterminated
    fences), balanced-object scanning that respects string literals and
    escapes, and trailing-comma repair.

    Args:
        text: Raw model output.

    Returns:
        The decoded JSON value.

    Raises:
        AgentError: If no JSON value could be recovered from ``text``.
    """
    if not isinstance(text, str) or not text.strip():
        raise AgentError("Model returned an empty response; no JSON value could be found.")

    direct = _try_load(text[:_MAX_SCAN])
    if direct is not _MISSING:
        return direct

    for body in _iter_fenced_bodies(text):
        parsed = _try_load(body)
        if parsed is not _MISSING:
            return parsed

    parsed = _try_balanced(text)
    if parsed is not _MISSING:
        return parsed

    snippet = text[:_ERROR_SNIPPET]
    raise AgentError(
        "Could not recover JSON from the model response. Reply with bare JSON only. "
        f"Response was: {snippet!r}"
    )


def _try_load(candidate: str) -> Any:
    """Parse ``candidate``, repairing trailing commas; ``_MISSING`` on failure."""
    if not candidate or not candidate.strip():
        return _MISSING
    text = _strip_unterminated_fence(candidate.strip())
    if len(text) > _MAX_SCAN:
        text = text[:_MAX_SCAN]
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        pass
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", text))
    except (ValueError, RecursionError):
        return _MISSING


def _strip_unterminated_fence(text: str) -> str:
    """Drop a trailing, unclosed markdown fence from ``text``."""
    if not text.endswith(_FENCE):
        return text
    body = text[: -len(_FENCE)].rstrip()
    if _FENCE in body:
        return text
    return body


def _iter_fenced_bodies(text: str) -> Iterator[str]:
    """Yield the body of every fenced block in ``text``, outermost first."""
    index = 0
    seen = 0
    while seen < _MAX_FENCES:
        start = text.find(_FENCE, index)
        if start < 0:
            return
        newline = text.find("\n", start)
        if newline < 0:
            body = text[start + len(_FENCE) :]
            if body.strip():
                yield body
            return
        body_start = newline + 1
        end = text.find(_FENCE, body_start)
        if end < 0:
            body = text[body_start : body_start + _MAX_SCAN]
            if body.strip():
                yield body
            return
        body = text[body_start:end]
        if body.strip():
            yield body
        index = end + len(_FENCE)
        seen += 1


def _try_balanced(text: str) -> Any:
    """Scan for the first balanced ``{``/``[`` region and parse it."""
    scanned = text[:_MAX_SCAN]
    starts = 0
    for index, char in enumerate(scanned):
        if char not in "{[":
            continue
        starts += 1
        if starts > _MAX_STARTS:
            return _MISSING
        found = _find_balanced(scanned, index)
        if found is None:
            continue
        parsed = _try_load(found)
        if parsed is not _MISSING:
            return parsed
    return _MISSING


def _find_balanced(text: str, start: int) -> str | None:
    """Return the balanced JSON region opening at ``text[start]``.

    Tracks nesting depth plus string/escape state so a brace inside a string
    literal does not terminate the region early. Returns ``None`` when the
    region is never closed within :data:`_MAX_SCAN` characters.
    """
    openers = "{["
    closers = "}]"
    depth = 0
    in_string = False
    escaped = False
    limit = min(len(text), start + _MAX_SCAN)
    for index in range(start, limit):
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
        elif char in openers:
            depth += 1
        elif char in closers:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
            if depth < 0:
                return None
    return None