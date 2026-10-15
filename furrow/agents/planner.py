from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=config or settings)

    async def plan(self, goal: str, task_history: list[TaskModel] | None = None) -> Plan:
        history_context = ""
        if task_history:
            completed = [t for t in task_history if t.status == "completed"]
            failed = [t for t in task_history if t.status == "failed"]
            if completed:
                history_context += "\n\nCompleted tasks:\n" + "\n".join(
                    f"- {t.id}: {t.description} ({t.result})" for t in completed
                )
            if failed:
                history_context += "\n\nFailed tasks (need fixing):\n" + "\n".join(
                    f"- {t.id}: {t.description} - {t.result}" for t in failed
                )

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{history_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
