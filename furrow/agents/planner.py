from __future__ import annotations

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        # Use custom planner prompt if provided, otherwise use default
        planner_prompt = getattr(self.settings, "planner_prompt", None) or PLANNER_PROMPT
        prompt = planner_prompt + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        max_retries = max(2, getattr(self.settings, "max_retries", 3))
        
        last_error = None
        for attempt in range(max_retries):
            response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response)
            except ValueError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Progressively stricter instructions
                    if attempt == 0:
                        prompt += (
                            "\n\nYour previous reply could not be parsed. "
                            "Reply with ONLY a JSON object matching the required shape, "
                            "no prose, no markdown fences."
                        )
                    else:
                        prompt += (
                            "\n\nCRITICAL: You MUST output ONLY valid JSON. "
                            "No explanations, no markdown, no extra text. "
                            "The JSON must have 'tasks' array and 'rationale' string."
                        )
                continue
        
        # All retries exhausted
        raise ValueError(
            f"Planner output could not be parsed after {max_retries} attempts. "
            f"Last error: {last_error}. Response: {str(response)[:500]}"
        ) from last_error
