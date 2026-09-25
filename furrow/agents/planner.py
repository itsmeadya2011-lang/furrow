from __future__ import annotations

import json

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None, cycle_history: list[dict] | None = None) -> Plan:
        existing_tasks = existing_tasks or []
        cycle_history = cycle_history or []
        
        # Build context from existing tasks and cycle history
        context_parts = []
        if existing_tasks:
            completed = [t for t in existing_tasks if t.status == "completed"]
            failed = [t for t in existing_tasks if t.status == "failed"]
            pending = [t for t in existing_tasks if t.status == "pending"]
            
            if completed:
                context_parts.append("Completed tasks:")
                for t in completed:
                    context_parts.append(f"  - {t.id}: {t.description} ({t.result[:100] if t.result else 'no result'})")
            if failed:
                context_parts.append("Failed tasks (need fixing):")
                for t in failed:
                    context_parts.append(f"  - {t.id}: {t.description} - {t.result}")
            if pending:
                context_parts.append("Pending tasks:")
                for t in pending:
                    context_parts.append(f"  - {t.id}: {t.description}")
        
        if cycle_history:
            context_parts.append("\nRecent cycle history:")
            for cycle in cycle_history[-3:]:  # Last 3 cycles
                context_parts.append(f"  Cycle {cycle['cycle']}: {len(cycle['tasks'])} tasks, tests {'passed' if cycle['test_result']['passed'] else 'failed'}")
        
        context = "\n".join(context_parts) if context_parts else "No previous tasks."
        
        prompt = f"{PLANNER_PROMPT}\n\nContext:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
