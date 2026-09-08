from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    """Agent that breaks a goal into parallelizable tasks."""

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        """Generate a plan (list of tasks) for the given goal."""
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        logger.info("Planning", goal=goal)
        response = await self._call_llm(prompt)

        plan = self._parse_plan(response, goal)
        logger.info("Plan parsed", tasks=len(plan.tasks))
        return plan

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_llm(self, prompt: str) -> str:
        return await self.client.complete(
            prompt,
            model=self.client.settings.planner_model,
        )

    def _parse_plan(self, response: str, goal: str) -> Plan:
        """Parse the LLM response into a Plan, with fallbacks for malformed output."""
        # Try direct JSON parse first
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try extracting JSON from markdown code block
        cleaned = self._extract_json_block(response)
        if cleaned:
            try:
                data = json.loads(cleaned)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: if the response mentions the goal is complete, return empty plan
        if "complete" in response.lower() and "no tasks" in response.lower():
            logger.info("Planner indicated goal is complete")
            return Plan(tasks=[], rationale="Goal is already complete according to planner.")

        # Final fallback: raise with context
        raise ValueError(
            f"Failed to parse plan from LLM.\n"
            f"Goal: {goal}\n"
            f"Response: {response[:2000]}"
        )

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        """Extract a JSON object from a markdown code block or raw text."""
        import re

        # Try markdown code block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        # Try raw JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return None
