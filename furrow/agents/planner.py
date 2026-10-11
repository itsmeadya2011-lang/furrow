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
        context = ""
        if existing_tasks:
            completed = [t for t in existing_tasks if t.status == "completed"]
            failed = [t for t in existing_tasks if t.status == "failed"]
            pending = [t for t in existing_tasks if t.status == "pending"]
            context += "\nExisting tasks:\n"
            if completed:
                context += f"  Completed: {', '.join(t.id for t in completed)}\n"
            if pending:
                context += f"  Pending: {', '.join(t.id for t in pending)}\n"
            if failed:
                context += f"  Failed: {', '.join(f'{t.id}: {t.result}' for t in failed)}\n"

        # Add codebase context
        try:
            files = self.client.list_files(self.client.settings.workspace)
            if files:
                context += "\nCodebase files:\n" + "\n".join(f"  {f}" for f in files[:50])
                if len(files) > 50:
                    context += f"\n  ... and {len(files) - 50} more files"
        except Exception:
            pass

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
