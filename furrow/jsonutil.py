from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["JSONExtractionError", "strip_code_fence", "extract_json", "extract_json_object"]


class JSONExtractionError(ValueError):
    """Raised when no parseable JSON can be extracted from a string."""


_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*$")


def strip_code_fence(text: str) -> str:
    """Remove leading and trailing markdown code fences from ``text``.

    Handles ``json`` info-strings, unterminated fences, and returns the input
    unchanged when no fence is present.

    Example::

        strip_code_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    """
    if not text:
        return text
    lines = text.splitlines()
    # Drop a leading fence line.
    if lines and _FENCE_RE.match(lines[0]):
        lines = lines[1:]
    # Drop a trailing fence line, tolerating a missing one.
    while lines and _FENCE_RE.match(lines[-1]):
        lines.pop()
    if not lines:
        return ""
    # If the last line is an unterminated fence, drop it.
    if lines[-1].lstrip().startswith("```"):
        lines.pop()
    return "\n".join(lines)


def _find_balanced(text: str, open_ch: str, close_ch: str) -> int:
    """Return the index of the matching close bracket for ``open_ch`` at position 0.

    Respects string literals and backslash escapes.  Returns -1 when unbalanced.
    """
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if in_string:
            if char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]`` (tolerant LLM output)."""
    result = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            close = "}" if ch == "{" else "]"
            end = _find_balanced(text[i:], ch, close)
            if end == -1:
                result.append(text[i:])
                break
            block = text[i + 1 : i + end]
            block = re.sub(r",\s*([}\]])", r"\1", block)
            block = re.sub(r",\s*$", "", block)
            result.append(ch)
            result.append(block)
            result.append(close)
            i += end + 1
        else:
            result.append(ch)
            i += 1
    return "".join(result)


def extract_json(text: str) -> Any:
    """Extract and parse the first JSON value from ``text``.

    Order: (1) ``json.loads`` of the stripped text; (2) strip fences and retry;
    (3) locate the first balanced ``{...}``/``[...]`` region; (4) retry after
    removing trailing commas.

    Raises:
        JSONExtractionError: when nothing parseable is found.
    """
    if text is None:
        raise JSONExtractionError("no text to parse")
    candidates = [text.strip(), strip_code_fence(text).strip()]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        region = text[start:]
        end = _find_balanced(region, open_ch, close_ch)
        if end == -1:
            continue
        candidate = region[: end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(_strip_trailing_commas(candidate))
        except json.JSONDecodeError:
            pass
    raise JSONExtractionError("no JSON object or array found in text")


def extract_json_object(text: str) -> dict[str, Any]:
    """Like :func:`extract_json` but require a mapping result.

    Raises:
        JSONExtractionError: when nothing parseable is found or the result is
            not a mapping.
    """
    value = extract_json(text)
    if not isinstance(value, dict):
        raise JSONExtractionError("extracted JSON is not an object")
    return value