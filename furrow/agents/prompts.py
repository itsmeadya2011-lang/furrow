from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

PLANNER_PROMPT = """You are a planning agent for an autonomous coding system called Furrow.

Break the user's goal into 1-5 parallelizable tasks. Each task must be:
- Independent when possible
- Completable in 1-3 tool call rounds by a worker
- Specific enough that a worker can implement it without ambiguity

Return JSON only (no markdown, no explanation) with this shape:
{
  "tasks": [
    {
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": []
    }
  ],
  "rationale": "Brief explanation of the plan"
}

If the goal is too large for one cycle, say so in rationale and break it into \
the most critical first slice."""

WORKER_PROMPT = """You are a worker agent in an autonomous coding system called Furrow.

Your job is to implement the assigned task completely and concisely.

Rules:
- Work only on the assigned task. Do not refactor unrelated code.
- Make minimal, targeted changes.
- Return a concise summary of what you changed and any issues.
- Do not spawn subagents.
"""

TESTER_PROMPT = """You are a tester agent in an autonomous coding system called Furrow.

Given the goal and test output, determine if tests passed or failed.

Return JSON only (no markdown) with this shape:
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}

If tests failed, list each failure in the failures array."""

# Matches ```lang\n<body>``` fences, capturing the language tag and the body.
_FENCE_RE = re.compile(
    r"```[ \t]*(?P<lang>[A-Za-z0-9_+#.-]*)[ \t]*\r?\n?(?P<body>.*?)```",
    re.DOTALL,
)
_MAX_DEBUG_CHARS = 200


def _object_spans(text: str) -> Iterator[str]:
    """Yield each brace-balanced ``{...}`` span in ``text``, outermost first.

    Braces inside string literals are ignored, and a backslash escapes the
    next character so ``\\"`` does not terminate a string. Scanning stops at
    the first unbalanced ``{`` because nothing after it can be trusted.
    """
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end == -1:
            return
        yield text[start:end]
        search_from = end


def _as_object(snippet: str) -> dict | None:
    """Parse ``snippet`` and return it when it is a JSON object, else None."""
    try:
        parsed = json.loads(snippet)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _unparseable_error(text: str) -> ValueError:
    """Build a ValueError carrying a truncated excerpt of the model response."""
    excerpt = text if isinstance(text, str) else repr(text)
    excerpt = excerpt.strip()
    if len(excerpt) > _MAX_DEBUG_CHARS:
        excerpt = excerpt[:_MAX_DEBUG_CHARS] + "..."
    return ValueError(f"no JSON object found in model response: {excerpt!r}")


def extract_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Tolerates markdown code fences and surrounding prose. Raises ValueError
    when no valid JSON object can be found.
    """
    if not isinstance(text, str) or not text.strip():
        raise _unparseable_error(text if isinstance(text, str) else repr(text))

    # 1. Markdown fences: try the body of each fenced block, in source order.
    candidates = [
        match.group("body").strip()
        for match in _FENCE_RE.finditer(text)
        if match.group("body").strip()
    ]
    # 2. Otherwise (or as a last resort) scan the whole response for an object.
    candidates.append(text.strip())

    for candidate in candidates:
        for snippet in _object_spans(candidate):
            parsed = _as_object(snippet)
            if parsed is not None:
                return parsed
    raise _unparseable_error(text)
