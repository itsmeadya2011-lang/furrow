from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            # extract_json copes with ```json fences and prose around the payload,
            # which is what models actually emit.  It raises ValueError when there is
            # nothing parseable at all.
            data = extract_json(response)
            if not isinstance(data, dict):
                raise TypeError(
                    f"expected a JSON object with 'tasks' and 'rationale', "
                    f"got {type(data).__name__}"
                )
            return Plan(**data)
        except (TypeError, ValueError) as exc:
            # ValueError covers both extract_json failures and pydantic's
            # ValidationError for well-formed JSON with the wrong shape.
            raise ValueError(f"Failed to parse plan from LLM: {exc}\nResponse: {response}") from exc