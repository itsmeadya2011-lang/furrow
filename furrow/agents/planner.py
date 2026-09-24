from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def plan(self, goal: str) -> Plan:
        logger.info("planning_started", goal=goal[:100])

        # Get workspace context
        workspace_files = self.client.list_files(self.settings.workspace)
        context = f"Workspace files:\n{chr(10).join(workspace_files[:50])}" if workspace_files else "Workspace is empty"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_parsed", tasks=len(plan.tasks), rationale=plan.rationale[:100])
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
