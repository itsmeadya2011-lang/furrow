from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings as SettingsType


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings_obj: SettingsType | None = None) -> None:
        self.client = client or LLMClient(settings=settings_obj or settings)

    async def plan(self, goal: str, project_files: list[str] | None = None, existing_tasks: list[TaskModel] | None = None) -> Plan:
        project_files = project_files or []
        existing_tasks = existing_tasks or []
        
        # Build context about existing project state
        context_parts = []
        if project_files:
            context_parts.append("Existing project files:\n" + "\n".join(f"  - {f}" for f in project_files[:50]))
        if existing_tasks:
            context_parts.append("Previously planned tasks:")
            for t in existing_tasks:
                status = "✓" if t.status == "completed" else "✗" if t.status == "failed" else "⏳"
                context_parts.append(f"  {status} {t.id}: {t.description} (files: {', '.join(t.files) or 'any'})")
        
        context = "\n\n".join(context_parts) if context_parts else "No existing project context."
        
        prompt = f"{PLANNER_PROMPT}\n\nProject Context:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
