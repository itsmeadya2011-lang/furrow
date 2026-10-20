from __future__ import annotations

import json
import re
from typing import Any


class LLMParseError(ValueError):
    def __init__(self, raw: str, message: str) -> None:
        super().__init__(message)
        self.raw = raw


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = strip_code_fences(text)
    candidate = _find_balanced(stripped, "{", "}")
    if candidate is None:
        raise LLMParseError(text, f"no JSON object found in: {text[:200]}")
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMParseError(text, f"invalid JSON: {e}; snippet: {text[:200]}") from e
    if not isinstance(value, dict):
        raise LLMParseError(text, f"expected object, got {type(value).__name__}: {text[:200]}")
    return value


def extract_json_array(text: str) -> list[Any]:
    stripped = strip_code_fences(text)
    candidate = _find_balanced(stripped, "[", "]")
    if candidate is None:
        raise LLMParseError(text, f"no JSON array found in: {text[:200]}")
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMParseError(text, f"invalid JSON: {e}; snippet: {text[:200]}") from e
    if not isinstance(value, list):
        raise LLMParseError(text, f"expected array, got {type(value).__name__}: {text[:200]}")
    return value


def find_json_bool(text: str, key: str) -> bool | None:
    pattern = re.compile(rf"""['"]{re.escape(key)}['"]\s*:\s*(true|false)""", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).lower() == "true"
