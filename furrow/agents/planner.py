from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, max_retries: int = 3) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = ""
        for attempt in range(1, max_retries + 1):
            response = await self.client.complete(
                prompt, model=self.client.settings.planner_model
            )
            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "Plan parse failed (attempt %d/%d): %s", attempt, max_retries, e
                )
                if attempt >= max_retries:
                    break
                # Retry with a hint to return valid JSON
                prompt = (
                    f"{PLANNER_PROMPT}\n\n"
                    f"IMPORTANT: Your previous response was not valid JSON. "
                    f"Return ONLY valid JSON, no markdown.\n"
                    f"Goal: {goal}\n"
                )

        logger.error(
            "Failed to parse plan after %d attempts. Falling back to empty plan.",
            max_retries,
        )
        logger.error("Last response: %s", response)
        return Plan(tasks=[], rationale=f"Failed to generate a valid plan after {max_retries} attempts.")
