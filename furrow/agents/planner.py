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
        completed_tasks = [t for t in existing_tasks if t.status == "completed"]
        pending_tasks = [t for t in existing_tasks if t.status in ("pending", "failed")]

        completed_summary = ""
        if completed_tasks:
            completed_summary = "\n\nAlready completed tasks:\n" + "\n".join(
                f"  - {t.id}: {t.description}" for t in completed_tasks
            )

        pending_summary = ""
        if pending_tasks:
            pending_summary = "\n\nPending/failed tasks (do not re-plan these):\n" + "\n".join(
                f"  - {t.id}: {t.description} (status: {t.status})" for t in pending_tasks
            )

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{completed_summary}{pending_summary}\n"
        logger.debug("planning", goal=goal, existing_task_count=len(existing_tasks))
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_parsed", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
