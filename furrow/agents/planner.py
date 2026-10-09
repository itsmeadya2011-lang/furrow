from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1 or last < first:
            raise ValueError("No JSON object found in LLM response")
        text = text[first : last + 1]
    return text


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            json_text = extract_json(response)
            data = json.loads(json_text)
            for task in data.get("tasks", []):
                if "id" in task and not isinstance(task["id"], str):
                    task["id"] = str(task["id"])
            plan = Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

        if len(plan.tasks) > self.client.settings.max_parallel_tasks:
            plan.tasks = plan.tasks[: self.client.settings.max_parallel_tasks]

        return plan