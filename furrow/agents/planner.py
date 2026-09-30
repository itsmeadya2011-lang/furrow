from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @staticmethod
    def _extract_json(response: str) -> dict:
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        raise ValueError(f"Failed to extract JSON from LLM response.\nResponse: {response}")

    @staticmethod
    def _validate_plan_data(data: dict, response: str) -> None:
        if "tasks" not in data:
            raise ValueError(f"Missing required field 'tasks' in plan response.\nResponse: {response}")
        if not isinstance(data["tasks"], list):
            raise TypeError(
                f"Field 'tasks' must be a list in plan response.\nResponse: {response}"
            )
        if "rationale" not in data:
            raise ValueError(
                f"Missing required field 'rationale' in plan response.\nResponse: {response}"
            )
        if not isinstance(data["rationale"], str):
            raise TypeError(
                f"Field 'rationale' must be a string in plan response.\nResponse: {response}"
            )

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)

        try:
            data = self._extract_json(response)
            self._validate_plan_data(data, response)
            return Plan(**data)
        except Exception:  # noqa: BLE001, S110
            pass

        retry_prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nReturn valid JSON only, no markdown."
        retry_response = await self.client.complete(
            retry_prompt, model=self.client.settings.planner_model
        )

        data = self._extract_json(retry_response)
        self._validate_plan_data(data, retry_response)
        return Plan(**data)
