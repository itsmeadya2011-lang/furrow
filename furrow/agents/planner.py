from __future__ import annotations

import json
from pathlib import Path
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
        # Get codebase context
        workspace = self.client.settings.workspace
        files = self.client.list_files(workspace)
        
        # Filter to relevant files (limit to avoid token overflow)
        relevant_files = [f for f in files if not f.startswith('.') and not any(f.startswith(d) for d in ['__pycache__', 'node_modules', '.git', 'dist', 'build'])]
        relevant_files = relevant_files[:100]  # Limit
        
        existing_context = ""
        if existing_tasks:
            completed = [t for t in existing_tasks if t.status == "completed"]
            failed = [t for t in existing_tasks if t.status == "failed"]
            pending = [t for t in existing_tasks if t.status == "pending"]
            
            existing_context = f"""
Existing tasks:
- Completed ({len(completed)}): {[t.id for t in completed]}
- Failed ({len(failed)}): {[t.id for t in failed]}
- Pending ({len(pending)}): {[t.id for t in pending]}
"""
        
        file_context = ""
        if relevant_files:
            file_context = f"\nExisting files in workspace:\n" + "\n".join(f"  {f}" for f in relevant_files)
        
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n{existing_context}{file_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
