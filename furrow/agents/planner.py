from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            # Validate the plan structure
            if not isinstance(data.get("tasks"), list):
                raise ValueError("Plan must contain a 'tasks' list")
            for i, task in enumerate(data["tasks"]):
                if not isinstance(task.get("id"), str):
                    raise ValueError(f"Task {i} must have a string 'id'")
                if not isinstance(task.get("description"), str):
                    raise ValueError(f"Task {i} must have a string 'description'")
            plan = Plan(**data)
            logger.info("plan_parsed", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:500])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
