from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT, extract_json_object
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlanParseError(ValueError):
    """Raised when the planner model output cannot be parsed into a Plan.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep working.
    """


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = extract_json_object(response)
        except ValueError as e:
            raise PlanParseError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e

        if not isinstance(data, dict):
            raise PlanParseError(
                f"Plan JSON must be an object, got {type(data).__name__}.\n"
                f"Response: {response}"
            )
        if "tasks" not in data:
            raise PlanParseError(f"Plan JSON is missing the 'tasks' key.\nResponse: {response}")
        tasks = data["tasks"]
        if not isinstance(tasks, list):
            raise PlanParseError(
                f"Plan JSON 'tasks' must be a list, got {type(tasks).__name__}.\n"
                f"Response: {response}"
            )

        try:
            return Plan(**data)
        except (TypeError, ValueError) as e:
            raise PlanParseError(f"Failed to parse plan from LLM: {e}\nResponse: {response}") from e