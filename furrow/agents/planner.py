from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path.cwd()

    def _get_codebase_context(self) -> str:
        """Get a summary of the codebase structure."""
        files = self.client.list_files(self.workspace)
        # Filter to relevant files (source code, config, etc.)
        relevant_extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.rs', '.go', '.java', '.cpp', '.c', '.h'}
        relevant_files = [f for f in files if Path(f).suffix in relevant_extensions or 'requirements' in f or 'package.json' in f or 'Cargo.toml' in f]
        return "\n".join(sorted(relevant_files)[:50])  # Limit to 50 files

    async def plan(self, goal: str, task_history: list[dict] | None = None) -> Plan:
        history_context = ""
        if task_history:
            history_context = "\n\nPrevious cycles:\n"
            for cycle in task_history[-3:]:  # Last 3 cycles
                history_context += f"Cycle {cycle['cycle']}: {cycle['plan_rationale']}\n"
                for task in cycle['tasks']:
                    history_context += f"  - {task['task_id']}: {task['status']}\n"
                    if task.get('result'):
                        history_context += f"    Result: {task['result'][:200]}\n"

        codebase_context = self._get_codebase_context()
        codebase_section = f"\n\nCodebase structure:\n{codebase_context}" if codebase_context else ""

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{codebase_section}{history_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
