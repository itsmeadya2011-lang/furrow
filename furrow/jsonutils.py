from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, default: Any = None) -> Any:
    """Extract a JSON object or array from arbitrary LLM output.

    Handles:
      - Plain JSON
      - Markdown code fences (``` or ```json)
      - Prose surrounding the JSON

    Raises ValueError if no valid JSON is found and no default is supplied.
    """
    if not text or not text.strip():
        if default is not None:
            return default
        raise ValueError("Empty response; cannot extract JSON")

    candidate = text.strip()

    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    if not candidate.startswith(("{", "[")):
        start = None
        for opener in ("{", "["):
            idx = candidate.find(opener)
            if idx != -1 and (start is None or idx < start):
                start = idx
        if start is None:
            if default is not None:
                return default
            raise ValueError("No JSON object or array found in response")
        candidate = candidate[start:]

    # Trim a trailing non-JSON suffix after the matching close bracket.
    candidate = _trim_trailing(candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        if default is not None:
            return default
        snippet = candidate[:200]
        raise ValueError(f"Failed to parse JSON: {snippet!r}")


def _trim_trailing(candidate: str) -> str:
    opener = candidate[0]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    end = None
    for i, ch in enumerate(candidate):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is not None:
        return candidate[:end]
    return candidate
