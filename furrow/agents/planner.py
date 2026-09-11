from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _extract_json(response: str) -> str:
    """Strip Markdown code fences and extract the outer JSON object.

    Handles responses like ````json {...} ```` or prose such as
    "Here is the plan: {...} hope that helps". The outermost braces are
    located by scanning with a depth counter so nested braces inside the
    JSON itself do not confuse the extraction.
    """
    text = response.strip()

    # Strip a leading code fence such as ```json or ``` and any trailing ```.
    if text.startswith("```"):
        # Remove the opening fence line (e.g. ```json).
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1 :]
        else:
            text = ""
        # Remove a trailing fence if present.
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
        text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unbalanced braces: no matching closing brace found")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        max_attempts = max(1, self.client.settings.max_attempts)
        last_response = ""
        for attempt in range(1, max_attempts + 1):
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            last_response = response
            try:
                data = json.loads(_extract_json(response))
                return Plan(**data)
            except (json.JSONDecodeError, ValueError):
                if attempt >= max_attempts:
                    raise ValueError(
                        f"Failed to parse plan from LLM after {max_attempts} attempts\n"
                        f"Response: {last_response}"
                    )
                continue

        raise ValueError(f"Failed to parse plan from LLM after {max_attempts} attempts\nResponse: {last_response}")