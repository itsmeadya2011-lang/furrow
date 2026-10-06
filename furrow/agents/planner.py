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
        existing_tasks = existing_tasks or []
        
        # Get project context
        project_files = self._get_project_context()
        completed_tasks = [t for t in existing_tasks if t.status == "completed"]
        failed_tasks = [t for t in existing_tasks if t.status == "failed"]
        pending_tasks = [t for t in existing_tasks if t.status == "pending"]

        context_parts = []
        if project_files:
            context_parts.append(f"Project files:\n{project_files}")
        if completed_tasks:
            context_parts.append(f"Completed tasks:\n" + "\n".join(f"- {t.id}: {t.description}" for t in completed_tasks))
        if failed_tasks:
            context_parts.append(f"Failed tasks (need fixing):\n" + "\n".join(f"- {t.id}: {t.description} - {t.result}" for t in failed_tasks))
        if pending_tasks:
            context_parts.append(f"Pending tasks:\n" + "\n".join(f"- {t.id}: {t.description}" for t in pending_tasks))

        context = "\n\n".join(context_parts) if context_parts else "No previous context."

        prompt = f"{PLANNER_PROMPT}\n\nContext:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    def _get_project_context(self) -> str:
        """Get a summary of project files for context."""
        try:
            files = self.client.list_files(Path.cwd())
            # Filter to relevant files (source code, config, tests)
            relevant = [f for f in files if any(f.endswith(ext) for ext in ('.py', '.js', '.ts', '.json', '.toml', '.yaml', '.yml', '.md', '.txt'))]
            # Limit to avoid token bloat
            return "\n".join(relevant[:50])
        except Exception:
            return ""
