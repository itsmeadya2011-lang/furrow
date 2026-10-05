from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient
from furrow.util import parse_json_object

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = parse_json_object(response)
            tasks_raw = data.get("tasks")
            if not isinstance(tasks_raw, list):
                raise ValueError(f"Plan response missing 'tasks' list. Response: {response[:200]}")
            tasks: list[TaskModel] = []
            for index, task in enumerate(tasks_raw):
                if not isinstance(task, dict):
                    raise ValueError(f"Plan response missing 'tasks' list. Response: {response[:200]}")
                task = dict(task)
                if not task.get("id"):
                    task["id"] = str(index + 1)
                tasks.append(TaskModel(**task))
            data["tasks"] = tasks
            data.setdefault("rationale", "")
            return Plan(**data)
        except ValueError as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e