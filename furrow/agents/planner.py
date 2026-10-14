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

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None, cycle: int = 1) -> Plan:
        # Get codebase context
        files = self.client.list_files(self.client.settings.workspace)
        file_tree = "\n".join(files[:100])  # Limit to first 100 files
        
        # Get completed task summaries
        completed_tasks = [t for t in (existing_tasks or []) if t.status == "completed"]
        completed_summary = "\n".join([f"- {t.id}: {t.description} -> {t.result[:100] if t.result else 'done'}" for t in completed_tasks])
        
        # Get failed tasks
        failed_tasks = [t for t in (existing_tasks or []) if t.status == "failed"]
        failed_summary = "\n".join([f"- {t.id}: {t.description} -> FAILED: {t.result[:200] if t.result else 'unknown'}" for t in failed_tasks])
        
        context = f"""Codebase files:
{file_tree}

Completed tasks this session:
{completed_summary if completed_summary else "  (none)"}

Failed tasks:
{failed_summary if failed_summary else "  (none)"}

Current cycle: {cycle}"""

        prompt = f"{PLANNER_PROMPT}\n\nContext:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
