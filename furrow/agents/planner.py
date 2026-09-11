from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing_tasks = existing_tasks or []
        completed = [t for t in existing_tasks if t.status == "completed"]
        failed = [t for t in existing_tasks if t.status == "failed"]
        pending = [t for t in existing_tasks if t.status == "pending"]
        
        context = ""
        if completed:
            context += "\nCompleted tasks:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in completed)
        if failed:
            context += "\nFailed tasks:\n" + "\n".join(f"  - {t.id}: {t.description} (error: {t.result})" for t in failed)
        if pending:
            context += "\nPending tasks:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in pending)
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n{context}\n"
        logger.debug("planning", goal=goal, existing_tasks=len(existing_tasks))
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_created", tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
