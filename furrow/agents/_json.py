import re


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        if text.endswith("```"):
            text = text[: text.rfind("```")].strip()
    return text
