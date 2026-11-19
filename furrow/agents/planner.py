from __future__ import annotations

import os
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

    def _get_project_context(self, workspace: Path | None = None) -> str:
        """Gather project context for better planning."""
        ws = workspace or Path(self.settings.workspace)
        parts = []
        
        # Project type detection
        markers = {
            "pyproject.toml": "Python (pyproject.toml)",
            "package.json": "Node.js",
            "Cargo.toml": "Rust",
            "go.mod": "Go",
            "pom.xml": "Java (Maven)",
            "build.gradle": "Java (Gradle)",
            "composer.json": "PHP",
            "requirements.txt": "Python (requirements.txt)",
        }
        for marker, desc in markers.items():
            if (ws / marker).exists():
                parts.append(f"Project type: {desc}")
                break
        
        # List top-level directories and key files
        try:
            entries = []
            for entry in ws.iterdir():
                if entry.name.startswith(".") or entry.name in {"__pycache__", "node_modules", "target", "build", "dist"}:
                    continue
                if entry.is_dir():
                    entries.append(f"  {entry.name}/")
                else:
                    entries.append(f"  {entry.name}")
            if entries:
                parts.append("Project structure:\n" + "\n".join(entries[:30]))
        except Exception:
            pass
        
        return "\n\n".join(parts) if parts else "No project context available."

    async def plan(self, goal: str, context: str = "", workspace: Path | None = None) -> Plan:
        project_context = self._get_project_context(workspace)
        
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if project_context:
            prompt += "\n\nProject Context:\n" + project_context
        if context:
            prompt += "\n\nContext:\n" + context

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
