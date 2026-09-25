from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing_tasks = existing_tasks or []
        # Get workspace files for context
        workspace_files = self.client.list_files(self.client.settings.workspace)

        # Build context about existing tasks
        task_context = ""
        if existing_tasks:
            task_context = "\nExisting tasks:\n"
            for t in existing_tasks:
                task_context += f"  - {t.id}: {t.description} (status: {t.status})\n"

        file_context = ""
        if workspace_files:
            file_context = "\nWorkspace files:\n" + "\n".join(
                f"  {f}" for f in workspace_files[:50]
            )

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{task_context}{file_context}\n"
        logger.debug(
            "planner_prompt",
            goal=goal,
            existing_tasks=len(existing_tasks),
            files=len(workspace_files),
        )
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_generated", tasks=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
