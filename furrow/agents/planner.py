from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

SYSTEM_PROMPT = (
    "You are a planning engine for an autonomous coding system. "
    "Respond with raw JSON only — no markdown fences, no prose."
)

MAX_ATTEMPTS = 3


def strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ``` fence and surrounding whitespace.

    Handles fences with or without a language tag, prose before/after the
    fence, and plain bare JSON (no fence is a no-op).
    """
    stripped = text.strip()
    lines = stripped.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].lstrip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _scan_object(text: str, start: int) -> dict | None:
    """Try to parse a balanced JSON object starting at `start`.

    Respects string literals and escapes so braces inside strings don't
    confuse the brace counter. Returns None if this start doesn't yield a
    valid JSON object.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        return None
                    if isinstance(data, dict):
                        return data
                    return None
    return None


def extract_json_object(text: str) -> dict:
    """Locate and parse the outermost JSON object embedded in `text`.

    Strips code fences first, then walks brace candidates (so prose before
    the JSON doesn't break us) with full string/escape awareness. Raises
    ValueError with a clear message if no JSON object is found.
    """
    cleaned = strip_code_fences(text)
    start = cleaned.find("{")
    while start != -1:
        data = _scan_object(cleaned, start)
        if data is not None:
            return data
        start = cleaned.find("{", start + 1)
    raise ValueError("no JSON object found in LLM response")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        cap = max(5, self.client.settings.max_parallel_tasks)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await self.client.complete(
                    prompt,
                    system=SYSTEM_PROMPT,
                    model=self.client.settings.planner_model,
                )
                return self._build_plan(extract_json_object(response), cap)
            except ValueError as e:
                snippet = response[:200]
                if attempt < MAX_ATTEMPTS - 1:
                    prompt = (
                        f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\n"
                        f"Your previous response was invalid: {e}\n"
                        f"Malformed output:\n{snippet}\n\n"
                        "Respond with raw JSON only (no markdown, no prose)."
                    )
                else:
                    raise ValueError(
                        f"Failed to parse plan from LLM after {MAX_ATTEMPTS} "
                        f"attempts: {e}\nLast response: {snippet}"
                    ) from e
        raise ValueError("unreachable")

    def _build_plan(self, data: dict, cap: int) -> Plan:
        # ValueError (not TypeError) on purpose: plan() retries on ValueError, so a
        # structurally wrong payload is fed back to the model instead of sinking the cycle.
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raise ValueError("'tasks' must be a list")  # noqa: TRY004

        tasks: list[TaskModel] = []
        for i, t in enumerate(raw_tasks):
            if not isinstance(t, dict):
                raise ValueError("each task must be an object")  # noqa: TRY004
            tid = (t.get("id") or "").strip() or str(i + 1)
            desc = (t.get("description") or "").strip()
            tasks.append(
                TaskModel(
                    id=tid,
                    description=desc,
                    files=t.get("files", []),
                    dependencies=t.get("dependencies", []),
                )
            )

        rationale = (data.get("rationale") or "").strip()
        if len(tasks) > cap:
            tasks = tasks[:cap]
            rationale = f"{rationale}\n[truncated to {cap} tasks]".strip()

        return Plan(tasks=tasks, rationale=rationale)
