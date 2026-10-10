from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, task_history: list[TaskModel] | None = None, workspace: Path | None = None) -> Plan:
        context = ""
        if workspace:
            files = self.client.list_files(workspace)
            if files:
                context += "\n\nExisting files in workspace:\n" + "\n".join(f"  {f}" for f in files[:50])
                if len(files) > 50:
                    context += f"\n  ... and {len(files) - 50} more files"

        if task_history:
            completed = [t for t in task_history if t.status == "completed"]
            failed = [t for t in task_history if t.status == "failed"]
            if completed:
                context += f"\n\nPreviously completed tasks ({len(completed)}):"
                for t in completed[-10:]:
                    context += f"\n  - {t.id}: {t.description}"
            if failed:
                context += f"\n\nPreviously failed tasks ({len(failed)}):"
                for t in failed[-5:]:
                    context += f"\n  - {t.id}: {t.description} (error: {t.result})"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        logger.debug("planning", goal=goal, context_length=len(context))
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_generated", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
