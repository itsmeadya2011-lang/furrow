from __future__ import annotations

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)

# Maximum number of attempts to obtain a parseable plan from the LLM.
# A transient malformed-LLM-output event must not kill an entire run.
MAX_PLANNER_ATTEMPTS = 4

# Each retry after the first appends an increasingly strict instruction.
_RETRY_INSTRUCTIONS = (
    "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
    "matching the required shape, no prose, no markdown fences.",
    "\n\nYour previous reply could not be parsed. Output exactly one JSON object "
    'with keys "tasks" (a list) and "rationale" (a string). Nothing else.',
    "\n\nYour previous reply could not be parsed. Reply with valid, bare JSON only: "
    '{"tasks": [...], "rationale": "..."}. No prose, no fences, no code blocks.',
)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        last_error: ValueError | None = None
        response: str = ""

        for attempt in range(1, MAX_PLANNER_ATTEMPTS + 1):
            if attempt > 1:
                prompt += _RETRY_INSTRUCTIONS[attempt - 2]
            response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response)
            except ValueError as exc:
                last_error = exc
                continue

        raw = response if isinstance(response, str) else str(response)
        logger.warning(
            "Planner output could not be parsed after %d attempts; "
            "falling back to an empty plan (goal treated as complete). "
            "Last error: %s. Response: %s",
            MAX_PLANNER_ATTEMPTS,
            last_error,
            raw[:500],
        )
        return Plan(tasks=[], rationale="Planner output unparseable; goal treated as complete.")
