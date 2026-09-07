from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, previous_context: str = "") -> Plan:
        context_block = f"Previous cycle results:\n{previous_context}\n\n" if previous_context else ""
        prompt = f"{PLANNER_PROMPT}\n\n{context_block}Goal: {goal}\n"

        last_error: str | None = None
        last_response: str = ""
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            if last_error:
                prompt = (
                    f"{PLANNER_PROMPT}\n\n{context_block}Goal: {goal}\n\n"
                    f"[Retry {attempt}/{max_attempts}] Previous attempt failed. "
                    f"Respond with valid JSON only. Do not include explanations or markdown fences.\n"
                )

            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            last_response = response

            try:
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = str(e)

        raise ValueError(
            f"Failed to parse plan from LLM after {max_attempts} attempts. "
            f"Last error: {last_error}\nRaw response: {last_response}"
        )
