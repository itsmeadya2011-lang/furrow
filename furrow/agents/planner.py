from __future__ import annotations

import json
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        project_structure = self._format_project_structure()
        prompt = f"{PLANNER_PROMPT.format(project_structure=project_structure)}\n\nGoal: {goal}\n"
        logger.info("planner_started", goal=goal)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("planner_completed", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planner_failed", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    def _format_project_structure(self) -> str:
        try:
            files = self.client.list_files(self.client.settings.workspace)
            if not files:
                return "(empty workspace)"
            return "\n".join(files[:50])
        except Exception:
            return "(could not list workspace files)"
