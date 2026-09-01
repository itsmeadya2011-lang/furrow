from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict | list | None:
    """Extract a JSON object or array from a string.

    Handles:
    - Plain JSON
    - JSON wrapped in markdown code fences (```json ... ```)
    - JSON embedded in prose (first balanced {…} or […] block)
    - Empty string -> None
    """
    if not text:
        return None

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    text = text.strip()
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first balanced JSON object or array
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return None
