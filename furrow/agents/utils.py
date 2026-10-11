from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Extract a JSON object from text that may contain markdown fences or prose.

    - Strips markdown code fences (```json ... ``` or ``` ... ```)
    - Tries to find the first ``{`` and last ``}`` and extracts that substring
    - Falls back to the full text when no JSON object can be identified
    """
    if not text:
        return text

    # Strip a markdown code fence if present (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1)

    # Extract the substring from the first { to the last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text