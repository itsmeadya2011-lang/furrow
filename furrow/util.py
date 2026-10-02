from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "JSONExtractionError",
    "PathEscapeError",
    "extract_json",
    "extract_json_object",
    "resolve_within",
    "truncate",
]

_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)
_SNIPPET_CHARS = 200


class JSONExtractionError(ValueError):
    """Raised when a JSON document cannot be recovered from model output."""


class PathEscapeError(ValueError):
    """Raised when a path resolves outside of its permitted root."""


def _loads(candidate: str) -> Any:
    return json.loads(candidate)


def _scan_balanced(text: str, start: int) -> int:
    """Return the index just past the container opened at ``start``, or -1 if unbalanced."""
    opening = text[start]
    closing = "}" if opening == "{" else "]"
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
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _first_container(text: str) -> Any:
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        end = _scan_balanced(text, index)
        if end == -1:
            continue
        try:
            return _loads(text[index:end])
        except ValueError:
            continue
    snippet = truncate(text, limit=_SNIPPET_CHARS, marker=" ... ")
    raise JSONExtractionError(f"no balanced JSON object or array found in: {snippet!r}")


def extract_json(text: str) -> Any:
    """Pull a JSON value out of an LLM reply.

    Tries the whole reply, then any fenced code block, then the first balanced ``{...}``/``[...]``
    container. Raises :class:`JSONExtractionError` when nothing parses.
    """
    if not isinstance(text, str):
        raise JSONExtractionError(f"expected str, got {type(text).__name__}")

    stripped = text.strip()
    if stripped:
        try:
            return _loads(stripped)
        except ValueError:
            pass

    for match in _FENCE_RE.finditer(text):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            return _loads(block)
        except ValueError:
            continue

    if stripped:
        return _first_container(stripped)

    raise JSONExtractionError("empty response")


def extract_json_object(text: str) -> dict[str, Any]:
    """Like :func:`extract_json` but requires a mapping at the top level."""
    data = extract_json(text)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    snippet = truncate(text, limit=_SNIPPET_CHARS, marker=" ... ")
    raise JSONExtractionError(
        f"expected a JSON object, got {type(data).__name__}: {snippet!r}"
    )


def truncate(text: str, limit: int = 2000, marker: str = "\n... [truncated] ...\n") -> str:
    """Keep the head and tail of ``text`` so both ends of a failure message survive.

    The result never exceeds ``limit``; when ``limit`` is too small to fit the marker, the
    leading ``limit`` characters are returned instead.
    """
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    budget = limit - len(marker)
    if budget < 2:
        return text[:limit]
    head = budget // 2 + budget % 2
    tail = budget - head
    return "".join([text[:head], marker, text[len(text) - tail :]])


def resolve_within(root: str | Path, candidate: str | Path) -> Path:
    """Resolve ``candidate`` (absolute, or relative to ``root``) and require it to stay inside."""
    root_path = Path(root).expanduser().resolve()
    target = Path(candidate).expanduser()
    if not target.is_absolute():
        target = root_path / target
    resolved = target.resolve()
    if not resolved.is_relative_to(root_path):
        raise PathEscapeError(f"{resolved} escapes root {root_path}")
    return resolved
