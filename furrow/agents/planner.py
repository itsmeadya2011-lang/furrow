from __future__ import annotations

from furrow.agents.parsing import extract_json_object
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = extract_json_object(response)
        except Exception as e:
            raise ValueError(
                f"Failed to parse plan from LLM: {e}\nResponse: {response[:200]}"
            ) from e
        tasks_raw = data.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise ValueError(f"Plan missing non-empty tasks: {response[:200]}")
        tasks: list[TaskModel] = []
        for i, t in enumerate(tasks_raw[:5], start=1):
            if not isinstance(t, dict) or not t.get("description"):
                continue
            tid = t.get("id")
            if not tid:
                tid = str(i)
            tasks.append(
                TaskModel(
                    id=str(tid),
                    description=str(t["description"]),
                    files=t.get("files") or [],
                    dependencies=t.get("dependencies") or [],
                )
            )
        if not tasks:
            raise ValueError(f"No valid tasks in plan: {response[:200]}")
        return Plan(tasks=tasks, rationale=str(data.get("rationale", "")))
