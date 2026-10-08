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

    async def plan(self, goal: str, pending_tasks: list[TaskModel] | None = None) -> Plan:
        # Build context from pending/failed tasks
        context_parts = []
        if pending_tasks:
            context_parts.append("Pending/failed tasks from previous cycles:")
            for task in pending_tasks:
                context_parts.append(f"  - {task.id}: {task.description} (status: {task.status})")
                if task.result:
                    context_parts.append(f"    Result: {task.result[:200]}")
        context = "\n".join(context_parts) if context_parts else "No previous tasks."

        # Get workspace file listing
        workspace_files = self.client.list_files(self.client.settings.workspace)
        workspace_list = "\n".join(f"  {f}" for f in workspace_files[:50])
        if len(workspace_files) > 50:
            workspace_list += f"\n  ... and {len(workspace_files) - 50} more files"

        prompt = PLANNER_PROMPT.format(
            context=context,
            workspace_files=workspace_list,
        ) + f"\n\nGoal: {goal}\n"

        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")