from __future__ import annotations

from furrow.agents.parsing import extract_json, parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        # First pass: get reasoning + plan
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            return parse_plan(response)
        except ValueError as first:
            # Second pass: stricter prompt asking for JSON only
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    async def plan_with_reasoning(self, goal: str, context: str = "") -> tuple[str, Plan]:
        """Plan with explicit chain-of-thought reasoning.
        
        Returns a tuple of (reasoning_text, Plan).
        The reasoning is extracted from the LLM response before the JSON.
        """
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        
        # Extract reasoning (everything before the JSON object)
        reasoning = ""
        try:
            json_str = extract_json(response)
            # Find where the JSON starts in the response
            json_start = response.find(json_str)
            if json_start > 0:
                reasoning = response[:json_start].strip()
        except ValueError:
            # If we can't extract JSON, treat the whole response as reasoning
            reasoning = response.strip()
        
        # Now parse the plan (reusing existing logic)
        try:
            plan = parse_plan(response)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                plan = parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second
        
        return reasoning, plan
