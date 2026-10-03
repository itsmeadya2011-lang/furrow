from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# ```json ... ``` (or ``` ... ```) wrappers around a JSON payload.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
# A fence line that is never closed, e.g. a response truncated mid-block.
_ORPHAN_FENCE_RE = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", re.MULTILINE)

_RETRY_HINT = (
    "Your previous response was not valid JSON. Return ONLY a raw JSON object, "
    "no markdown fences, no prose."
)
# Never re-ask more than this many extra times, whatever the settings say.
_MAX_PLAN_RETRIES = 3


def _json_candidates(text: str) -> Iterator[str]:
    """Yield progressively more aggressive views of ``text`` to try to parse."""
    yield text
    unfenced = _FENCE_RE.sub(lambda match: match.group(1), text)
    if unfenced != text:
        yield unfenced.strip()
    stripped = _ORPHAN_FENCE_RE.sub("", unfenced)
    if stripped != unfenced:
        yield stripped.strip()


def _first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces in string literals."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
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
                return text[start : index + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Parse a single JSON object out of an LLM response.

    Tolerates markdown fences and surrounding prose: anything that is not a bare
    object is searched for the first balanced ``{...}`` block.
    """
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty response from LLM")
    for view in _json_candidates(candidate):
        try:
            data = json.loads(view)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    block = _first_json_object(candidate)
    if block is None:
        raise ValueError(f"No JSON object found in response: {candidate!r}")
    data = json.loads(block)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
    return data


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings

    async def plan(self, goal: str) -> Plan:
        base_prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        extra_attempts = min(max(self.settings.max_retries, 0), _MAX_PLAN_RETRIES)
        response = ""
        last_error: Exception = ValueError("Planner produced no usable response.")
        for attempt in range(extra_attempts + 1):
            prompt = base_prompt if attempt == 0 else f"{base_prompt}\n{_RETRY_HINT}"
            response = await self.client.complete(prompt, model=self.settings.planner_model)
            if not response or not response.strip():
                last_error = ValueError("LLM returned an empty response.")
                continue
            try:
                return Plan(**extract_json(response))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise ValueError(
            f"Failed to parse plan from LLM: {last_error}\nResponse: {response}"
        ) from last_error
