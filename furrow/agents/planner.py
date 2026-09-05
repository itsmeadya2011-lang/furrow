from __future__ import annotations

import json
from typing import TYPE_CHECKING

from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()

        # Strip Markdown fences: ```json ... ``` or ~~~json ... ~~~
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop opening fence
            lines = lines[1:]
            # Drop closing fence
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        elif text.startswith("~~~"):
            lines = text.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("~~~"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Locate the first balanced { ... } JSON object
        depth = 0
        in_quote = False
        escape = False
        start = None
        for i, ch in enumerate(text):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_quote:
                escape = True
                continue
            if ch == '"':
                in_quote = not in_quote
                continue
            if in_quote:
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    json_str = text[start : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        preview = text[:200]
        raise ValueError(f"Could not locate valid JSON object in LLM response. Preview: {preview!r}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=4), reraise=True)
    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = self._extract_json(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse plan from LLM after retries. Last response (500 chars): {response[:500]!r}"
            ) from e
