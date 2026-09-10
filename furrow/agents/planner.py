from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, pending_tasks: list | None = None) -> Plan:
        context = ""
        if pending_tasks:
            context = "\n\nPreviously planned tasks (preserve completed ones):\n"
            for task in pending_tasks:
                context += f"- {task.id}: {task.description} (status: {task.status})\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("plan_generated", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log.error("plan_parse_failed", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")