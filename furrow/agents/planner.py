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

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        existing_tasks = existing_tasks or []
        
        # Build context about existing tasks
        context = ""
        if existing_tasks:
            completed = [t for t in existing_tasks if t.status == "completed"]
            failed = [t for t in existing_tasks if t.status == "failed"]
            pending = [t for t in existing_tasks if t.status == "pending"]
            
            context = f"\nExisting tasks ({len(existing_tasks)} total):"
            if completed:
                context += f"\n  Completed ({len(completed)}): " + ", ".join(t.id for t in completed)
            if failed:
                context += f"\n  Failed ({len(failed)}): " + ", ".join(f"{t.id} ({t.result})" for t in failed)
            if pending:
                context += f"\n  Pending ({len(pending)}): " + ", ".join(t.id for t in pending)
        
        # Get codebase file listing for context
        file_list = self.client.list_files(self.client.settings.workspace)
        if file_list:
            context += f"\n\nProject files ({len(file_list)}):\n" + "\n".join(f"  {f}" for f in file_list[:50])
            if len(file_list) > 50:
                context += f"\n  ... and {len(file_list) - 50} more files"

        prompt = f"{PLANNER_PROMPT}\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
