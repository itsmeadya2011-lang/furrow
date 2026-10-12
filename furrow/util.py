from __future__ import annotations

import json
import re
from typing import Any


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (e.g. ```json ... ```) from LLM output."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from LLM output, ignoring fences and prose.

    Raises ValueError if no JSON object can be found or parsed.
    """
    cleaned = strip_code_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, dict):
                return data
    raise ValueError(f"No JSON object found in response:\n{text}")