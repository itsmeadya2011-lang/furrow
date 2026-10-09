from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(response[start : end + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
            else:
                raise ValueError(f"Failed to parse plan from LLM: no JSON object found\nResponse: {response}")
        if isinstance(data, dict) and "tasks" in data:
            plan = Plan(**data)
        else:
            raise ValueError(f"Failed to parse plan from LLM: missing 'tasks' key\nResponse: {response}")
        limit = max(1, self.client.settings.max_parallel_tasks)
        plan.tasks = plan.tasks[:limit]
        return plan
