from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_MAX_SNIPPET = 500


def _truncate(text: str, limit: int = _MAX_SNIPPET) -> str:
    """Return a safely-sized snippet of *text* for error messages."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_json(text: str) -> str:
    """Extract the JSON value from LLM output that may be wrapped in a markdown
    fence or surrounded by prose.

    LLMs frequently emit ```` ```json ... ``` ```` fences, bare ```` ``` ... ``` ````
    fences, or a JSON value buried in a sentence of commentary.  Naive regex
    extraction fails on any of these, and the planner/tester then raise and kill
    the cycle.  This function locates the first ``{`` or ``[`` and brace-matches
    through the string (honouring string literals and escapes) to find the
    matching close, so nested braces inside string values do not confuse it.

    Raises:
        ValueError: if no JSON value can be located.  The original text is
            included (truncated) so failures are diagnosable.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Cannot extract JSON from empty or non-string input: {_truncate(text)!r}")

    # Strip a leading/trailing markdown code fence if present.  We look for the
    # first fence opener and, if one exists, cut everything up to and including
    # its newline, and strip a matching trailing fence line.
    stripped = text.strip()
    fence_open = None
    for marker in ("```json", "```JSON", "```"):
        if stripped.startswith(marker):
            fence_open = marker
            break
    if fence_open is not None:
        # Drop the opening fence line (and any language tag on it).
        newline = stripped.find("\n")
        if newline == -1:
            # Only a fence, no content.
            raise ValueError(f"No JSON value found after fence: {_truncate(text)!r}")
        stripped = stripped[newline + 1 :].strip()
        # Drop a trailing fence line if present.
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()

    start = -1
    open_ch = ""
    close_ch = ""
    for i, ch in enumerate(stripped):
        if ch in "{[":
            start = i
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start == -1:
        raise ValueError(f"No JSON value found in response: {_truncate(text)!r}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    raise ValueError(f"Unbalanced JSON value in response: {_truncate(text)!r}")


def parse_json_model(text: str, model: type[T]) -> T:
    """Extract the JSON from *text* and validate it into a pydantic *model*.

    Combines :func:`extract_json` with model construction so callers do not
    repeat the fence-prose-handling logic.  On any failure a ``ValueError`` is
    raised whose message contains the offending text (truncated) so the failure
    is diagnosable without re-prompting the LLM blindly.

    Raises:
        ValueError: if the text contains no JSON, is not valid JSON, or does not
            match the model schema.
    """
    try:
        raw = extract_json(text)
        data = json.loads(raw)
    # json.JSONDecodeError subclasses ValueError, so the narrower handler has to
    # come first; the bare `except ValueError: raise` after it is what lets the
    # other ValueErrors from extract_json keep their original message.
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}. Text: {_truncate(text)!r}") from e
    except ValueError:
        raise
    try:
        return model(**data)
    except Exception as e:  # pydantic.ValidationError and friends
        raise ValueError(
            f"Response did not match {model.__name__}: {e}. Text: {_truncate(text)!r}"
        ) from e
