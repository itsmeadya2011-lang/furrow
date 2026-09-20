from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings
        if self.settings is None:
            self.settings = getattr(client, "settings", None) or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str | None = None) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\n<goal>\n{goal}\n</goal>\n"
        if context is not None:
            prompt += f"<context>\n{context}\n</context>\n"

        last_error: Exception | None = None
        responses: list[str] = []
        for _ in range(2):
            response = await self.client.complete(prompt, model=self.settings.planner_model)
            responses.append(response)
            try:
                data = json.loads(response)
                return Plan.model_validate(data)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc

        raise ValueError(
            "Failed to parse and validate a plan after 2 attempts.\n"
            f"Attempt 1 response: {responses[0]!r}\n"
            f"Attempt 2 response: {responses[1]!r}\n"
            f"Response: {responses[-1]!r}\n"
            f"Last error: {last_error}"
        ) from last_error
