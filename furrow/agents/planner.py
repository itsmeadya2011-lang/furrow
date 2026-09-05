from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.base import Agent
from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)


class PlannerAgent(Agent):
    """Planner agent that breaks a goal into parallelizable tasks.

    Uses the LLM to produce a structured :class:`~furrow.config.Plan`
    with tasks, their target files, and inter-task dependencies.
    """

    system_prompt = PLANNER_PROMPT
    tools = []  # Planner doesn't need tools, it just returns structured JSON

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        super().__init__(client=client, settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{self.system_prompt}\n\nGoal: {goal}\n"
        model = self.client.settings.planner_model
        response = await self.client.complete(prompt, model=model)

        # Strip markdown code fences if the LLM wrapped the JSON
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove the opening ``` and closing ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
