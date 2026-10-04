"""Planner agent: turns a goal into an ordered, dependency-aware :class:`Plan`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from furrow.agents.jsonutil import JSONExtractionError, extract_json_object, extract_json_value
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    """Asks the LLM for a plan and validates it into a :class:`Plan`.

    The agent owns no state beyond its client, and its ``settings`` always come from
    that client so there is a single source of truth for configuration.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        # ``LLMClient`` resolves ``settings=None`` to ``get_settings()``, so passing
        # None through explicitly is safe. (Assigning ``client or LLMClient(settings)``
        # and then reading ``self.client.settings`` used to break when a caller passed
        # ``settings=None``, because the old client stored None verbatim.)
        self.client = client or LLMClient(settings=settings)

    @property
    def settings(self) -> Settings:
        """The settings used for planning, read from the client."""
        return self.client.settings

    async def plan(self, goal: str) -> Plan:
        """Return an execution plan for ``goal``.

        Raises:
            ValueError: if the model reply contains no JSON object (a
                :class:`~furrow.agents.jsonutil.JSONExtractionError`) or if the JSON
                does not match the :class:`Plan` schema.
        """
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(
            prompt,
            model=self.settings.planner_model,
            system=PLANNER_SYSTEM,
        )
        data = self._parse(response)
        try:
            return Plan.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                f"Failed to parse plan from LLM: {exc}\nResponse: {response[:2000]}"
            ) from exc

    @staticmethod
    def _parse(response: str) -> dict[str, Any]:
        """Extract the plan object from a model reply.

        Tolerates code fences, leading prose and trailing commentary. A reply that is a
        bare JSON array of tasks is accepted too, since that is a common shape.
        """
        try:
            return extract_json_object(response)
        except JSONExtractionError as exc:
            value = extract_json_value(response)
            if isinstance(value, list):
                return {"tasks": value}
            raise ValueError(f"Failed to parse plan from LLM: {exc}") from exc
