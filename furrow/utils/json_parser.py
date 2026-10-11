from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON object from text, handling markdown code blocks and extra text."""
    # Try to find JSON in markdown code blocks
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object directly
    # Look for the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


def parse_llm_response(text: str, model_class: type[T]) -> T | None:
    """Parse LLM response into a Pydantic model."""
    data = extract_json(text)
    if data is None:
        return None
    try:
        return model_class(**data)
    except Exception:
        return None


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default