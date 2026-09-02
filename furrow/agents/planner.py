from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_RETRIES = 3


def validate_plan_structure(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Plan must be a JSON object")
    if "tasks" not in data:
        raise ValueError("Plan missing required 'tasks' field")
    if not isinstance(data["tasks"], list):
        raise ValueError("'tasks' must be an array")
    for i, task in enumerate(data["tasks"]):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {i} must be an object")
        if "id" not in task:
            raise ValueError(f"Task at index {i} missing required 'id' field")
        if "description" not in task:
            raise ValueError(f"Task at index {i} missing required 'description' field")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_error: str | None = None

        for attempt in range(MAX_RETRIES):
            if last_error:
                prompt = (
                    f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\n"
                    f"Previous attempt failed with error: {last_error}\n"
                    f"Please correct the response and return valid JSON.\n"
                )

            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            try:
                data = json.loads(response)
                validate_plan_structure(data)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = f"{e}\nResponse: {response}"

        return Plan(
            tasks=[],
            rationale=f"Failed to generate valid plan after {MAX_RETRIES} attempts. Last error: {last_error}",
        )
