from __future__ import annotations

import json
from typing import Any

__all__ = ["extract_json"]

_FENCE_LANGUAGES = {"", "json", "json5", "jsonc"}


def _strip_code_fence(text: str) -> str | None:
    """Return the inner body when ``text`` is a single fenced code block, else ``None``."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None
    newline = stripped.find("\n")
    if newline == -1:
        return None
    # The opening fence is ``` optionally followed by an info string.
    info = stripped[3:newline].strip().lower()
    if info not in _FENCE_LANGUAGES:
        return None
    body = stripped[newline + 1 :]
    if not body.rstrip().endswith("```"):
        return None
    return body.rstrip()[: -len("```")].strip()


def _scan_balanced(text: str) -> str | None:
    """Return the first balanced ``{...}``/``[...]`` block, ignoring braces inside strings."""
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        closing = "}" if char == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current in "{[":
                depth += 1
            elif current in "}]":
                depth -= 1
                if depth == 0:
                    if current != closing:
                        break
                    return text[start : index + 1]
    return None


def extract_json(text: str) -> Any:
    """Extract and parse the first JSON value from raw model output.

    Handles: bare JSON, ```json ... ``` / ``` ... ``` fenced blocks,
    and JSON embedded in surrounding prose. Returns the parsed value.
    Raises ValueError with the offending text included if nothing parses.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected str, got {type(text).__name__}")

    source = text.strip()
    if not source:
        raise ValueError("No JSON found in empty response")

    candidates: list[str] = []
    fenced = _strip_code_fence(source)
    if fenced is not None:
        candidates.append(fenced)
    candidates.append(source)

    for candidate in candidates:
        if not candidate:
            continue
        # Bare JSON first, so a valid whole-string document wins over a scan.
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        block = _scan_balanced(candidate)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    snippet = text if len(text) <= 200 else text[:200] + "..."
    raise ValueError(f"No valid JSON found in response: {snippet!r}")
