from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def plan(self, goal: str, previous_tasks: list[TaskModel] | None = None) -> Plan:
        previous_context = ""
        if previous_tasks:
            completed = [t for t in previous_tasks if t.status == "completed"]
            failed = [t for t in previous_tasks if t.status == "failed"]
            if completed:
                previous_context += "\n\nPreviously completed tasks:\n"
                for t in completed:
                    previous_context += f"  - {t.id}: {t.description}\n"
            if failed:
                previous_context += "\nPreviously failed tasks:\n"
                for t in failed:
                    previous_context += f"  - {t.id}: {t.description} (error: {t.result})\n"
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{previous_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_generated", num_tasks=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:500])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
