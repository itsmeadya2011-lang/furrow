import json
import re


def parse_json_response(response: str) -> dict | list:
    cleaned = response.strip()

    fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    fences = fence_pattern.findall(cleaned)
    if fences:
        cleaned = "\n".join(fences).strip()

    if not cleaned:
        raise ValueError(f"No JSON content found in response: {response}")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for i, char in enumerate(cleaned):
        if char not in "{[":\n            continue
        start = i
        open_char = char
        close_char = "}" if char == "{" else "]"
        depth = 1
        for j in range(i + 1, len(cleaned)):
            c = cleaned[j]
            if c == open_char:
                depth += 1
            elif c == close_char:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : j + 1].strip()
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        continue

    raise ValueError(f"Failed to parse JSON from response: {response}")
