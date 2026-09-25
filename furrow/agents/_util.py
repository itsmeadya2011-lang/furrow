from __future__ import annotations

import json
from typing import Any


def parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse a JSON object out of an LLM response.

    Tolerates markdown fences and trailing commentary by locating the first
    ``{`` and last ``}`` if direct parsing fails.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return None
    return None
