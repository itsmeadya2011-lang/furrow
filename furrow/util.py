from __future__ import annotations

import json
import re


def strip_code_fences(text: str) -> str:
    """Strip a leading/trailing markdown code fence, returning inner content."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*\n?", text)
    if match:
        start = match.end()
        end = len(text)
        if text.rstrip().endswith("```"):
            end = text.rstrip().rfind("```")
            trailing = text[end:].rstrip()
            if trailing == "```":
                end = text.rstrip().rfind("```")
        return text[start:end].strip()
    return text


def parse_json_object(text: str) -> dict:
    """Robust JSON object extraction for LLM output."""
    text = strip_code_fences(text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
            raise ValueError(f"No JSON object found in response: {text[:200]}")
        try:
            result = json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"No JSON object found in response: {text[:200]}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object, got {type(result).__name__}")
    return result
