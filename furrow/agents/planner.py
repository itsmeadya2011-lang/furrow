from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        # Get file listing for context
        workspace = self.settings.workspace
        files = self.client.list_files(workspace)
        files_context = ""
        if files:
            files_context = f"\n\nProject files:\n" + "\n".join(f"  {f}" for f in files[:50])
            if len(files) > 50:
                files_context += f"\n  ... and {len(files) - 50} more files"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{files_context}"
        if context:
            prompt += f"\n\n{context}"
        logger.debug("Planning request", goal=goal, context_len=len(context), files_count=len(files))
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("Plan parsed", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse plan", error=str(e), response=response[:500])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
