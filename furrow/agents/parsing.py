from __future__ import annotations

import json
import re
from typing import Any, Iterator, Optional

# ```json ... ``` or ``` ... ``` blocks, with or without a language tag.
_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n?(.*?)```", re.IGNORECASE | re.DOTALL)

_OPENERS = {"{": "}", "[": "]"}
_CLOSERS = frozenset("}]")

# Sentinel for "nothing decoded yet"; ``None`` is a legitimate JSON result.
_UNSET: Any = object()


def extract_json(text: str) -> Any:
    """Recover the first JSON value from a raw LLM response.

    Models rarely emit a bare JSON document: they wrap it in markdown fences
    (``` ```json ``` blocks) and add prose before or after it, neither of which
    ``json.loads`` tolerates. This helper tolerates both, in order:

    1. the contents of each fenced code block, outermost text last;
    2. the response as-is, stripped;
    3. every balanced ``{...}``/``[...]`` span, matching braces while skipping JSON
       string literals and backslash escapes.

    Prefer a decoded value that carries content: a bare ``null``/``{}``/``[]`` is
    only returned when nothing more substantial parses, so prose that merely
    mentions an empty container cannot win over the real payload.

    Returns:
        The decoded value (dict, list, or scalar).

    Raises:
        ValueError: If no JSON value can be recovered. A ``json.JSONDecodeError``
            from the closest candidate is attached as ``__cause__``.
    """
    first: Any = _UNSET
    last_error: Optional[ValueError] = None
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError as exc:
            last_error = exc
            continue
        if value is not None and value != {} and value != []:
            return value
        if first is _UNSET:
            first = value
    if first is not _UNSET:
        return first
    raise ValueError(f"No JSON value found in response: {text[:200]!r}") from last_error


def _candidates(text: str) -> Iterator[str]:
    """Yield progressively looser views of ``text``, likeliest to parse first."""
    for segment in (*_fence_bodies(text), text):
        stripped = segment.strip()
        if stripped:
            yield stripped
        for index, char in enumerate(segment):
            if char not in _OPENERS:
                continue
            end = _matching_index(segment, index)
            if end != -1:
                yield segment[index : end + 1]


def _fence_bodies(text: str) -> list[str]:
    """Return the body of every markdown code fence in ``text``."""
    return [match.group(1) for match in _FENCE_RE.finditer(text)]


def _matching_index(text: str, start: int) -> int:
    """Index of the bracket closing the opener at ``start``, or ``-1`` if unbalanced.

    String literals and escapes are skipped so brackets inside JSON string values
    do not corrupt the depth count.
    """
    stack: list[str] = []
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
        elif char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in _CLOSERS:
            if not stack or stack.pop() != char:
                return -1
            if not stack:
                return index
    return -1
