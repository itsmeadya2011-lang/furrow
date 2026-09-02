from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    """Breaks a high-level goal into parallelizable, independent tasks.

    Uses an LLM to generate a plan. If the first attempt fails to produce
    valid JSON, it retries with a stricter prompt before raising.
    """

    MAX_RETRIES: int = 2

    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        """Generate a plan for the given goal.

        Args:
            goal: The high-level goal to plan for.

        Returns:
            A parsed Plan object with tasks.

        Raises:
            ValueError: If the LLM response cannot be parsed after retries.
        """
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"

        last_error: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            response = await self.client.complete(
                prompt, model=self.client.settings.planner_model
            )
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    # Retry with stricter instructions
                    prompt = (
                        f"{PLANNER_PROMPT}\n\n"
                        f"IMPORTANT: You must return ONLY valid JSON. No markdown, no explanations. "
                        f"Ensure every field matches the schema exactly.\n\n"
                        f"Goal: {goal}\n"
                    )

        # All retries exhausted
        detail = f"Failed to parse plan after {self.MAX_RETRIES} attempts."
        if last_error:
            detail += f" Last error: {last_error}"
        raise ValueError(f"{detail}\nResponse: {response}")
