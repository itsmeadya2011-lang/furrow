from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FENCE_RE = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences so raw JSON can be parsed."""
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        data = self._parse(response)
        return self._normalize(data)

    def _parse(self, response: str) -> dict:
        text = _strip_fences(response)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
        if not isinstance(data, dict) or "tasks" not in data:
            raise ValueError(f"Invalid plan structure: missing 'tasks'. Response: {response}")
        return data

    def _normalize(self, data: dict) -> Plan:
        raw_tasks = data.get("tasks") or []
        seen: set[str] = set()
        normalized: list[TaskModel] = []
        for i, t in enumerate(raw_tasks):
            if not isinstance(t, dict):
                continue
            desc = t.get("description")
            if not desc:
                continue
            tid = str(t.get("id") or i + 1)
            if tid in seen:
                tid = f"{tid}-{i}"
            seen.add(tid)
            normalized.append(
                TaskModel(
                    id=tid,
                    description=str(desc),
                    files=list(t.get("files") or []),
                    dependencies=list(t.get("dependencies") or []),
                )
            )
        return Plan(tasks=normalized, rationale=str(data.get("rationale", "")))