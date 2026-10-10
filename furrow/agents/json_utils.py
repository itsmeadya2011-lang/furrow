from __future__ import annotations

import json
import re


def parse_json_from_llm(text: str) -> dict:
    """Parse JSON from LLM output, handling markdown fences and extracting the first balanced block."""
    cleaned = text.strip()
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    fence_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```\s*$', cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract first balanced {...} or [...] block
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    break

    raise ValueError(f"Failed to parse JSON from LLM response: {text[:200]}")