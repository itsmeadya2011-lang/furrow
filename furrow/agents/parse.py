"""Helpers for extracting JSON from LLM responses.

LLMs frequently wrap JSON output in markdown fences (e.g. ````json ... ````)
or surround it with explanatory prose. This module provides a small,
dependency-free helper to recover the JSON substring before parsing.
"""

from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Return the first JSON object or array substring from ``text``.

    The function is tolerant of markdown-wrapped responses:

    * If a fenced code block is present (```json ... ``` or ``` ... ```), the
      content of the first such block is returned.
    * Otherwise, the first ``{`` or ``[`` is located and the matching closing
      ``}`` or ``]`` is used to slice the JSON span.
    * If no ``{`` or ``[`` exists, ``ValueError`` is raised.
    """
    # Look for a fenced code block first.
    fence_match = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    start_idx = -1
    open_char = ""
    close_char = ""
    for i, ch in enumerate(text):
        if ch == "{":
            start_idx = i
            open_char = "{"
            close_char = "}"
            break
        if ch == "[":
            start_idx = i
            open_char = "["
            close_char = "]"
            break

    if start_idx == -1:
        raise ValueError("No JSON found in response")

    # Find the matching closing bracket by scanning from the start, skipping
    # over nested structures and quoted strings so we don't stop early.
    depth = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        ch = text[i]
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
        elif ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : i + 1]

    raise ValueError("No JSON found in response")