from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


def _list_project_files(workspace: Path, limit: int = 50) -> list[str]:
    """Return a list of project files for context, excluding common build/cache dirs."""
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv", "dist", "build", ".furrow"}
    files: list[str] = []
    try:
        for path in workspace.rglob("*"):
            if path.is_file():
                rel = path.relative_to(workspace)
                if not any(part in skip_dirs for part in rel.parts):
                    files.append(str(rel))
                    if len(files) >= limit:
                        break
    except Exception:
        pass
    return files


def _get_project_context(workspace: Path) -> str:
    """Build a context string describing the project structure."""
    files = _list_project_files(workspace)
    if not files:
        return "No project files found."
    
    # Group by extension
    by_ext: dict[str, list[str]] = {}
    for f in files:
        ext = Path(f).suffix or "no-ext"
        by_ext.setdefault(ext, []).append(f)
    
    parts = ["Project structure:"]
    for ext, fs in sorted(by_ext.items()):
        parts.append(f"  {ext}: {', '.join(fs[:10])}{'...' if len(fs) > 10 else ''}")
    return "\n".join(parts)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        # Build project context for better planning
        project_ctx = _get_project_context(Path(self.settings.workspace))
        
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext from previous cycle:\n" + context
        prompt += "\n\n" + project_ctx

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
