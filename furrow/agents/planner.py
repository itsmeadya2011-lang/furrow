from __future__ import annotations

import json

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import CycleState, Plan, Settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def plan(self, goal: str, history: list[CycleState] | None = None) -> Plan:
        history_context = ""
        if history:
            history_context = "\n\nPrevious cycles:\n"
            for cycle in history[-3:]:  # Last 3 cycles
                history_context += f"Cycle {cycle.cycle_number}: {cycle.goal[:200]}\n"
                for task in cycle.tasks:
                    history_context += f"  Task {task.id}: {task.description[:100]} [{task.status}]\n"
                if cycle.test_result:
                    history_context += f"  Tests: {'PASS' if cycle.test_result.passed else 'FAIL'} - {cycle.test_result.summary}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{history_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")