from __future__ import annotations

import json

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger("furrow.planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        # Get project context
        files = self.client.list_files(self.settings.workspace)
        file_list = "\n".join(files[:100]) if files else "No files found"

        # Format existing tasks for context
        existing_context = ""
        if existing_tasks:
            existing_context = "\n\nExisting tasks:\n"
            for t in existing_tasks:
                existing_context += f"  - {t.id}: {t.description} ({t.status})\n"

        prompt = f"{PLANNER_PROMPT}\n\nProject files:\n{file_list}\n\nGoal: {goal}{existing_context}\n"
        log.debug("planning", goal=goal[:100], num_files=len(files), existing_tasks=len(existing_tasks) if existing_tasks else 0)
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("plan_parsed", num_tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            log.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")