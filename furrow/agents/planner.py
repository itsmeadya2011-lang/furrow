from __future__ import annotations

import json
from typing import Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, pending_tasks: list[Any] | None = None) -> Plan:
        # Get project context
        project_context = await self._get_project_context()

        # Build pending tasks context
        pending_context = ""
        if pending_tasks:
            pending_context = "\n\nPending/In-progress tasks from previous cycles:\n"
            for task in pending_tasks:
                pending_context += f"  - {task.id}: {task.description} (status: {task.status})\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nProject Context:\n{project_context}{pending_context}"
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

        # Filter to relevant files (source code, config, etc.)
        relevant_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".cpp", ".c", ".h",
            ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".md", ".txt",
            ".html", ".css", ".scss", ".sass", ".vue", ".svelte",
        }

        relevant_files = [
            f for f in files
            if any(f.endswith(ext) for ext in relevant_extensions)
            and not any(skip in f for skip in [".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv"])
        ][:50]  # Limit to 50 files

        context = f"Project root: {workspace}\n"
        context += f"Relevant files ({len(relevant_files)}):\n"
        for f in relevant_files:
            context += f"  - {f}\n"

        # Read key files for more context
        key_files = ["README.md", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "setup.py", "requirements.txt"]
        for key_file in key_files:
            if key_file in relevant_files:
                try:
                    content = await self.client.read_file(workspace / key_file)
                    context += f"\n--- {key_file} ---\n{content[:2000]}"
                except OSError:
                    pass

        return context