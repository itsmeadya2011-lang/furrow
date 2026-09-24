from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from text, handling markdown code blocks and other formats."""
    # Try to find JSON in markdown code blocks
    code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find JSON object in the text
    json_pattern = r"(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})"
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try the whole text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return None


def parse_llm_response(text: str, model_class: type[T]) -> T | None:
    """Parse LLM response into a Pydantic model with fallback strategies."""
    data = extract_json(text)
    if data:
        try:
            return model_class(**data)
        except Exception:  # noqa: BLE001, S110
            pass
    return None