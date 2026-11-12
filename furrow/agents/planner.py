from __future__ import annotations

import asyncio
from pathlib import Path

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "", workspace: Path | None = None) -> Plan:
        # Gather codebase context if workspace is provided
        codebase_context = ""
        if workspace is not None:
            codebase_context = await self._gather_codebase_context(workspace)

        # Build the full prompt with all context
        prompt_parts = [PLANNER_PROMPT, f"\n\nGoal: {goal}"]
        if codebase_context:
            prompt_parts.append(f"\n\nCodebase Context:\n{codebase_context}")
        if context:
            prompt_parts.append(f"\n\nAdditional Context:\n{context}")

        prompt = "".join(prompt_parts)

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            return parse_plan(response)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    async def _gather_codebase_context(self, workspace: Path) -> str:
        """Gather relevant codebase context for planning."""
        parts = []

        # List top-level files and directories
        try:
            entries = await asyncio.to_thread(self.client.list_files, workspace)
            # Filter to relevant files (not hidden, not build artifacts)
            relevant = [e for e in entries if not e.startswith(".") and not e.startswith("__")]
            if relevant:
                parts.append("Project structure (top 100 entries):\n" + "\n".join(sorted(relevant)[:100]))
        except Exception:
            pass

        # Read key configuration files
        key_files = [
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "README.md",
            "README.rst",
            "setup.py",
            "requirements.txt",
        ]
        for key_file in key_files:
            path = workspace / key_file
            if path.exists():
                try:
                    content = await self.client.read_file(path)
                    parts.append(f"\n--- {key_file} ---\n{content[:3000]}")
                except Exception:
                    pass

        return "\n".join(parts) if parts else ""
