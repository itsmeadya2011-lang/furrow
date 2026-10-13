from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient
from .logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.logger = get_logger(__name__)

    async def plan(self, goal: str) -> Plan:
        self.logger.info("Starting planning", goal=goal)
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        for attempt in range(2):
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.error(f"Failed to parse plan from LLM: {e}")
                if attempt == 0:
                    response = await self.client.complete(
                        f"{prompt}\n\nThe previous response was not valid JSON. Please return only valid JSON matching the required schema.",
                        model=self.client.settings.planner_model,
                    )
                else:
                    return Plan(tasks=[TaskModel(id="1", description=goal, files=[], dependencies=[], status="pending")], rationale="Fallback plan due to invalid JSON response from LLM")
        return Plan(tasks=[TaskModel(id="1", description=goal, files=[], dependencies=[], status="pending")], rationale="Fallback plan due to invalid JSON response from LLM")