from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import TaskModel


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, existing_tasks: list[TaskModel] | None = None) -> Plan:
        context = await self._gather_context()
        existing = ""
        if existing_tasks:
            existing = f"\n\nExisting tasks from previous cycles:\n{json.dumps([t.model_dump() for t in existing_tasks], indent=2)}"

        prompt = f"{PLANNER_PROMPT}\n\nCodebase context:\n{context}\n\nGoal: {goal}{existing}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _gather_context(self) -> str:
        """Gather relevant codebase context for planning."""
        client = self.client
        workspace = client.settings.workspace

        # Get list of files
        files = client.list_files(workspace)
        if not files:
            return "No files found in workspace."

        # Filter to relevant files (source code, configs)
        relevant_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".cs", ".rb", ".php", ".json", ".yaml", ".yml", ".toml", ".md", ".txt"}
        relevant_files = [f for f in files if Path(f).suffix in relevant_extensions]

        # Read first 50 lines of up to 10 most relevant files
        context_parts = []
        for f in relevant_files[:10]:
            try:
                content = await client.read_file(workspace / f)
                lines = content.split("\n")[:50]
                context_parts.append(f"--- {f} ---\n" + "\n".join(lines))
            except Exception:  # noqa: BLE001
                continue

        return "\n\n".join(context_parts) if context_parts else "No readable source files found."