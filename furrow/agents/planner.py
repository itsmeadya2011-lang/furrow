from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing = existing_tasks or []
        completed = [t for t in existing if t.status == "completed"]
        failed = [t for t in existing if t.status == "failed"]
        pending = [t for t in existing if t.status == "pending"]
        
        context = ""
        if completed:
            context += "Already completed:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in completed) + "\n\n"
        if failed:
            context += "Failed (need fixing):\n" + "\n".join(f"  - {t.id}: {t.description} - {t.result}" for t in failed) + "\n\n"
        if pending:
            context += "Pending:\n" + "\n".join(f"  - {t.id}: {t.description}" for t in pending) + "\n\n"
        
        prompt = f"{PLANNER_PROMPT}\n\n{context}Goal: {goal}\n"
        if existing:
            prompt += "\nPlan the NEXT steps only. Do not re-plan completed tasks. Use new unique task IDs."
        
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
