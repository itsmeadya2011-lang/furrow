from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("furrow.planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        """Generate a plan by breaking the goal into parallelizable tasks."""
        log.info("planner.plan.start", goal=goal)

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"

        # Include workspace file listing as context
        files = self.client.list_files(self.client.settings.workspace)
        if files:
            file_listing = "\n".join(f"  - {f}" for f in files[:50])
            prompt += f"\nExisting files in workspace:\n{file_listing}\n"
            if len(files) > 50:
                prompt += f"\n... and {len(files) - 50} more files\n"

        response = await self.client.complete(prompt, model=self.client.settings.planner_model)

        try:
            data = json.loads(response)
            plan = Plan(**data)
            log.info("planner.plan.done", num_tasks=len(plan.tasks), rationale=len(plan.rationale))
            return plan
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.error("planner.plan.parse_error", error=str(e))
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from None


__all__ = ["PlannerAgent"]
