from __future__ import annotations

import json

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

# Fallback single-task plan used when the LLM fails to produce valid JSON.
_FALLBACK_TASK = {
    "id": "1",
    "description": "Implement the requested changes",
    "files": [],
    "dependencies": [],
}


class PlannerAgent:
    """Agent that breaks a goal into parallelizable tasks."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        # Include workspace context so the planner can reference existing files
        workspace = self.client.settings.workspace
        try:
            files = await self.client.list_files(workspace)
        except OSError:
            files = []

        files_summary = (
            f"Existing workspace files: {len(files)} files found"
            if files
            else "Workspace is empty or unreadable"
        )

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n{files_summary}\n\nOutput JSON:\n"

        try:
            data = await self.client.complete_json(prompt, model=self.client.settings.planner_model)
            tasks_data = data.get("tasks", [])
            if not tasks_data:
                raise ValueError("LLM returned no tasks")

            tasks = []
            for i, td in enumerate(tasks_data):
                tasks.append(
                    TaskModel(
                        id=str(td.get("id", i + 1)),
                        description=td.get("description", ""),
                        files=td.get("files", []),
                        dependencies=td.get("dependencies", []),
                    )
                )

            rationale = data.get("rationale", "No rationale provided")
            return Plan(tasks=tasks, rationale=rationale)

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return self._fallback_plan(goal, str(e))

    def _fallback_plan(self, goal: str, error: str) -> Plan:
        """Create a minimal fallback plan when LLM output is invalid."""
        task = TaskModel(**_FALLBACK_TASK)
        task.description = f"Implement: {goal}"
        return Plan(
            tasks=[task],
            rationale=(
                f"Fallback plan due to LLM parsing error: {error}. "
                f"Breaking goal into a single direct implementation task."
            ),
        )
