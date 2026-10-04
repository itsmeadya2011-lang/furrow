"""
Shared text helpers for the agent layer.

LLM output is prose, not data: models wrap JSON in markdown fences, prefix it with
"Here's the plan:", append a closing remark, or emit a fenced block with trailing
commentary. ``json.loads`` on the raw string raises ``JSONDecodeError`` for every one
of those shapes and takes the whole cycle down with it.

:func:`extract_json_object` recovers a JSON object from such replies using a
brace/string-escape-aware scanner instead of the naive
``text[text.index("{"):text.rindex("}") + 1]`` slice, which breaks on braces inside
string literals and on unbalanced braces in surrounding prose.

:func:`truncate_output` is the other half of the story: agent output (test logs, file
contents) is unbounded, and an over-long prompt is truncated by the LLM layer into a
confusing downstream error.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "JSONExtractionError",
    "extract_json_object",
    "extract_json_value",
    "strip_code_fences",
    "truncate_output",
]

# How much of the offending text is quoted back in the error message.
_ERROR_TEXT_LIMIT = 400
# Minimum number of characters to keep on each side when the middle is elided.
_MIN_EDGE = 200
DEFAULT_ELISION_MARKER = "\n\n[... {dropped} characters elided ...]\n\n"


class JSONExtractionError(ValueError):
    """Raised when no JSON value can be recovered from model output.

    Subclasses :class:`ValueError` so callers that already guard against malformed
    model replies keep working.
    """


_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+.-]*)[ \t]*\r?\n(?P<body>.*?)(?:```|\Z)",
    re.DOTALL,
)


def truncate_output(
    text: str,
    max_chars: int,
    marker: str = DEFAULT_ELISION_MARKER,
) -> str:
    """Shorten ``text`` to at most ``max_chars`` characters, keeping head and tail.

    Test-runner failures, stack traces and diffs are almost always at the *end* of the
    output, while the command banner is at the start, so the middle is what gets
    elided. Returns ``text`` unchanged when it already fits.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker_len = len(marker)
    if max_chars <= marker_len:
        return text[:max_chars]
    # Split the remaining budget evenly, favouring the tail by any odd character.
    remaining = max_chars - marker_len
    head = (remaining + 1) // 2
    tail = remaining - head
    if head < _MIN_EDGE or tail < _MIN_EDGE:
        # Too tight to keep both ends meaningful: keep the tail, which carries the
        # failure detail, and note the truncation.
        return marker + text[-(max_chars - marker_len) :]
    dropped = len(text) - head - tail
    return text[:head] + marker.format(dropped=dropped) + text[-tail:]


def strip_code_fences(text: str) -> str:
    """Return the body of the most likely code fence in ``text``.

    Prefers a fence tagged ``json``, then any other fence. Text without a complete
    fenced block is returned unchanged.
    """
    blocks = _fenced_blocks(text)
    if not blocks:
        return text
    for language, body in blocks:
        if "json" in language.lower():
            return body
    return blocks[0][1]


def extract_json_value(text: str, default: Any = None) -> Any:
    """Return the first JSON value found in ``text``, or ``default``.

    Unlike :func:`extract_json_object` this accepts any JSON type, which lets callers
    handle models that answer with a bare array. Note that a candidate that parses to
    a non-object does *not* stop the search: a later object is still preferred.
    """
    if not text or not text.strip():
        return default
    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    return default


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text``.

    Candidates are tried in order of how likely they are to be the real payload:

    1. the whole response (the well-behaved case),
    2. the body of a ``json``-tagged code fence,
    3. the body of any other code fence,
    4. every balanced ``{...}`` span in the text, outermost first.

    Brace matching is string- and escape-aware, so nested objects, braces inside
    string literals and escaped quotes all survive intact.

    Raises:
        JSONExtractionError: if no candidate parses to an object. The message quotes
            the offending text (truncated) so the failure is diagnosable.
    """
    if not text or not text.strip():
        raise JSONExtractionError("model returned an empty response; no JSON object to parse")

    # Non-object JSON is remembered so the error can say *why* nothing was returned.
    wrong_type: str | None = None
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
        if wrong_type is None:
            wrong_type = type(value).__name__
    if wrong_type is None:
        detail = "no code fence and no balanced '{...}' block"
    else:
        detail = f"parsed JSON was {wrong_type}, not an object"
    raise JSONExtractionError(
        f"no JSON object found in model response ({detail}): {_excerpt(text)}"
    )


# -- internals ------------------------------------------------------------


def _candidates(text: str) -> list[str]:
    """Return every substring of ``text`` that might be the JSON payload."""
    stripped = text.strip()
    candidates = [stripped]
    json_fences: list[str] = []
    other_fences: list[str] = []
    for language, body in _fenced_blocks(text):
        body = body.strip()
        if not body:
            continue
        if "json" in language.lower():
            json_fences.append(body)
        else:
            other_fences.append(body)
    candidates.extend(json_fences)
    candidates.extend(other_fences)
    for start, end in _iter_balanced_spans(text):
        span = text[start:end]
        if span not in candidates:
            candidates.append(span)
    return candidates


def _fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(language, body)`` for each markdown code fence in ``text``."""
    return [(m.group("lang") or "", m.group("body")) for m in _FENCE_RE.finditer(text)]


def _iter_balanced_spans(
    text: str,
    start: int = 0,
    stop: int | None = None,
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs of balanced ``{...}`` spans, outermost first.

    An opening brace that never closes does not stop the scan: the cursor advances past
    it so a later, complete object is still found. That is what makes
    ``"Set {'a': 1} then {\"tasks\": []}"`` recoverable.
    """
    limit = len(text) if stop is None else stop
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < limit:
        begin = text.find("{", cursor)
        if begin == -1 or begin >= limit:
            break
        end = _match_brace(text, begin, limit)
        if end == -1:
            cursor = begin + 1
            continue
        spans.append((begin, end))
        cursor = end
    return spans


def _match_brace(text: str, begin: int, limit: int) -> int:
    """Return the index just past the ``}`` closing the ``{`` at ``begin``, or -1.

    Tracks string literals and backslash escapes so that ``{"re": "\\{not a brace\\"}"}``
    and ``{"tpl": "{literal}"}`` are matched correctly.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(begin, limit):
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
                return index + 1
    return -1


def _excerpt(text: str, limit: int = _ERROR_TEXT_LIMIT) -> str:
    """Return a short, single-line-safe excerpt of ``text`` for error messages."""
    excerpt = text.strip()
    if len(excerpt) > limit:
        excerpt = excerpt[:limit] + "... [truncated]"
    return _quote(excerpt)


def _quote(value: str | None) -> str:
    """Render ``value`` for an error message."""
    if value is None:
        return "nothing"
    return repr(value)
