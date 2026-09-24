from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, project_files: list[str] | None = None, previous_tasks: list[dict] | None = None) -> Plan:
        project_files = project_files or []
        previous_tasks = previous_tasks or []
        
        # Build context from project files and previous tasks
        context_parts = []
        
        if project_files:
            nl = chr(10)
            project_lines = [f"  - {f}" for f in project_files[:50]]
            context_parts.append(f"Project files:\n{nl.join(project_lines)}")
        
        if previous_tasks:
            completed = [t for t in previous_tasks if t.get("status") == "completed"]
            failed = [t for t in previous_tasks if t.get("status") == "failed"]
            pending = [t for t in previous_tasks if t.get("status") == "pending"]
            
            nl = chr(10)
            if completed:
                completed_lines = [f"  - {t['id']}: {t['description']}" for t in completed]
                context_parts.append(f"Completed tasks:\n{nl.join(completed_lines)}")
            if failed:
                failed_lines = [f"  - {t['id']}: {t['description']} (error: {t.get('result', 'unknown')})" for t in failed]
                context_parts.append(f"Failed tasks:\n{nl.join(failed_lines)}")
            if pending:
                pending_lines = [f"  - {t['id']}: {t['description']}" for t in pending]
                context_parts.append(f"Pending tasks:\n{nl.join(pending_lines)}")
        
        context = "\n\n".join(context_parts) if context_parts else "No previous context."
        
        prompt = f"{PLANNER_PROMPT}\n\nContext:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")