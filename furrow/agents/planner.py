from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, history: list[TaskModel] | None = None) -> Plan:
        history = history or []
        
        # Build context from history
        completed_tasks = [t for t in history if t.status == "completed"]
        failed_tasks = [t for t in history if t.status == "failed"]
        
        context_parts = []
        if completed_tasks:
            context_parts.append("Completed tasks:")
            for t in completed_tasks:
                context_parts.append(f"  - {t.id}: {t.description} -> {t.result[:100] if t.result else 'done'}")
        
        if failed_tasks:
            context_parts.append("Failed tasks:")
            for t in failed_tasks:
                context_parts.append(f"  - {t.id}: {t.description} -> {t.result}")
        
        context = "\n".join(context_parts) if context_parts else "No previous tasks."
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nContext:\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
