from __future__ import annotations

import json
import re


def extract_json(text: str) -> str:
    """Extract the first JSON object or array from a possibly noisy LLM response.

    Handles markdown code fences (```json ... ``` or ``` ... ```) and leading
    or trailing prose. Returns the raw JSON substring.

    Raises ValueError if no JSON object or array can be found.
    """
    if not text or not text.strip():
        raise ValueError("Empty response")

    stripped = text.strip()

    # Fast path: the whole response is valid JSON.
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Strip code fences if present.
    fenced = re.search(r"```(?:json|JSON)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    # Fall back to the outermost {...} or [...] block with balanced braces.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for i in range(start, len(stripped)):
                ch = stripped[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = stripped[start : i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            break
            start = stripped.find(opener, start + 1)

    raise ValueError("No valid JSON object or array found in response")