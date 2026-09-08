from __future__ import annotations

import json
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _complete(self, prompt: str, model: str) -> str:
        return await self.client.complete(prompt, model=model)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        last_error = None

        for attempt in range(3):
            current_prompt = prompt if attempt == 0 else f"{prompt}\n\nPrevious response was not valid JSON. Please return ONLY valid JSON."
            try:
                response = await self._complete(current_prompt, model=self.client.settings.planner_model)
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                logger.warning("plan_parse_failed", attempt=attempt + 1, error=str(e))
                continue
            except Exception as e:
                last_error = e
                logger.error("plan_api_failed", attempt=attempt + 1, error=str(e))
                continue

        logger.error("plan_failed_after_retries", error=str(last_error))
        return Plan(tasks=[], rationale="Failed to parse plan after retries")
