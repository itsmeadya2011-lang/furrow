from __future__ import annotations

import json
from typing import Any


def parse_json_response(text: str) -> Any:
    """Extract and parse the first JSON object/array from an LLM response.

    LLMs habitually wrap JSON in markdown code fences (```json ... ```, a bare
    ```, or with a different language tag) or embed it in surrounding prose.
    This helper normalizes such responses so callers get a parsed value instead
    of a constant ``json.JSONDecodeError``.

    Strategy:
    1. Strip markdown code-fence *markers* (the ``` lines), keeping the content
       between them intact.
    2. Walk the cleaned text with a brace/bracket depth counter that is aware
       of string literals and ``\\`` escapes, so braces or brackets that appear
       inside strings never unbalance the count.
    3. Parse the first slice that balances back to depth zero. If that slice
       is not valid JSON (prose braces such as ``{curly}``), resume the scan
       just after it and try the next balanced slice. If nothing parseable is
       found, raise a :class:`ValueError` with a truncated snippet of the
       *original* text for debugging.

    Args:
        text: The raw LLM response, possibly fenced or prose-wrapped.

    Returns:
        The first complete JSON object or array found, parsed into Python.

    Raises:
        ValueError: If ``text`` is not a string, or if no parseable JSON
            object or array can be located. The message includes a truncated
            snippet of the original text.
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected a string response, got {type(text).__name__}")

    cleaned = _strip_markdown_fences(text)
    found = _find_json_value(cleaned)
    if found is None:
        raise ValueError(_format_parse_error(text))
    return found[0]


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown fenced-code-block markers, keeping inner content.

    Every line whose stripped form starts with ``` ``` ``` is dropped as a
    marker line, whether it opens or closes a block. The text between markers
    is preserved verbatim so embedded JSON remains parseable.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("```")
    )


def _find_json_value(text: str) -> tuple[Any, int] | None:
    """Find and parse the first parseable JSON value in ``text``.

    Uses a single depth counter: every ``{`` or ``[`` increments it and every
    matching ``}`` or ``]`` decrements it. Because valid JSON is properly
    nested, depth returns to zero exactly at the end of the top-level value.
    String literals and escape sequences are tracked so that brackets inside
    strings are ignored.

    A structurally balanced candidate is not necessarily valid JSON: prose
    such as ``Use {curly} braces like this: {"a": 1}`` balances ``{curly}``
    first. When a candidate fails ``json.loads`` the scan does not give up;
    it restarts just past that candidate with a fresh counter, so the depth,
    string and escape state can never be double-counted across the restart
    and the scan always moves forward. ``None`` therefore means *no
    parseable* value was found anywhere in the text, not merely that the
    first balanced group was invalid.

    Returns:
        A ``(parsed_value, end_index)`` tuple for the first balanced value
        that is also valid JSON, or ``None`` if no such value can be found.
    """
    resume_from = 0
    while True:
        start: int | None = None
        depth = 0
        in_string = False
        escape = False
        failed_candidate_end: int | None = None

        for i in range(resume_from, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if in_string:
                if ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                if start is None:
                    start = i
                depth += 1
            elif ch in "}]":
                if start is None:
                    # Stray closing bracket before any opening value; skip it.
                    continue
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate), i + 1
                    except json.JSONDecodeError:
                        # Balanced but not JSON (stray braces in prose). Drop
                        # the whole candidate and rescan from just after it.
                        failed_candidate_end = i + 1
                        break

        if failed_candidate_end is None:
            # No balanced group remains in the rest of the text.
            return None
        # Strictly greater than the previous resume point, so this terminates.
        resume_from = failed_candidate_end


def _format_parse_error(text: str, limit: int = 500) -> str:
    """Build a descriptive ValueError message with a truncated text snippet."""
    snippet = text if len(text) <= limit else text[:limit] + "...(truncated)"
    return (
        "No valid JSON object or array found in the LLM response.\n"
        f"Snippet: {snippet!r}"
    )
