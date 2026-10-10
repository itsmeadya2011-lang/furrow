from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()

    async def plan(self, goal: str) -> Plan:
        # Get project context
        files = self.client.list_files(self.settings.workspace)
        project_context = ""
        if files:
            project_context = f"\n\nProject files:\n" + "\n".join(f"  {f}" for f in files[:50])

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n{project_context}\n"
        logger.debug("planning", goal=goal, files_count=len(files))
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_parsed", tasks_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response)
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
