from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:[a-zA-Z]+)?\n(.*?)```", re.DOTALL)


def parse_json_response(response: str) -> dict:
    """Extract a JSON object from an LLM response.

    Handles responses that may include markdown code fences or surrounding
    prose by locating the first ``{...}`` block and parsing it as JSON.

    Raises:
        json.JSONDecodeError: if no valid JSON object is found.
    """
    text = (response or "").strip()

    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            raise

    if not isinstance(data, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, 0)
    return data
