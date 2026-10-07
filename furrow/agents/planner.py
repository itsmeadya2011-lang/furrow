from __future__ import annotations

import json
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

    async def plan(self, goal: str) -> Plan:
        # Get project context
        context = await self._get_project_context()
        
        prompt = f"{PLANNER_PROMPT}\n\nProject Context:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _get_project_context(self) -> str:
        """Get relevant project context for planning."""
        workspace = self.client.settings.workspace
        files = self.client.list_files(workspace)
        
        # Limit to relevant files (source code, config, etc.)
        relevant_extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}
        relevant_files = [
            f for f in files 
            if Path(f).suffix in relevant_extensions and not any(skip in f for skip in ['__pycache__', '.git', 'node_modules', 'venv', '.env'])
        ]
        
        context_parts = [f"Project files ({len(relevant_files)} relevant):"]
        for f in sorted(relevant_files)[:50]:  # Limit to 50 files
            context_parts.append(f"  {f}")
        
        return "\n".join(context_parts)
