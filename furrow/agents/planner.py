from __future__ import annotations

from pathlib import Path

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)

    async def plan(
        self,
        goal: str,
        context: str = "",
        history: list | None = None,
        existing_files: list[str] | None = None,
    ) -> Plan:
        # Build project context
        project_context = self._build_project_context(existing_files)
        
        # Build history context
        history_context = self._build_history_context(history)
        
        prompt = PLANNER_PROMPT
        if project_context:
            prompt += "\n\nProject Context:\n" + project_context
        if history_context:
            prompt += "\n\nRecent History:\n" + history_context
        prompt += "\n\nGoal: " + goal
        if context:
            prompt += "\n\nAdditional Context:\n" + context

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

    def _build_project_context(self, existing_files: list[str] | None) -> str:
        """Build context about the project structure and key files."""
        parts = []
        
        # List key project files
        key_files = [
            "pyproject.toml", "package.json", "Cargo.toml", "go.mod",
            "requirements.txt", "setup.py", "setup.cfg", "pytest.ini",
            "tsconfig.json", "vite.config.ts", "webpack.config.js",
        ]
        found_key_files = []
        for kf in key_files:
            if (self.workspace / kf).exists():
                found_key_files.append(kf)
        if found_key_files:
            parts.append(f"Key project files: {', '.join(found_key_files)}")
        
        # List top-level directories
        try:
            dirs = [d.name for d in self.workspace.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if dirs:
                parts.append(f"Top-level directories: {', '.join(sorted(dirs)[:20])}")
        except (OSError, PermissionError):
            pass
        
        # Include existing files if provided
        if existing_files:
            parts.append(f"Files in current plan: {', '.join(existing_files[:30])}")
        
        return "\n".join(parts)

    def _build_history_context(self, history: list | None) -> str:
        """Build context from recent execution history."""
        if not history:
            return ""
        
        parts = []
        for entry in history[-5:]:  # Last 5 cycles
            cycle = entry.get("cycle", "?")
            rationale = entry.get("plan_rationale", "")
            test_summary = entry.get("test_summary", "")
            tests_passed = entry.get("tests_passed")
            parts.append(
                f"Cycle {cycle}: {rationale[:100]} | Tests: {'passed' if tests_passed else 'failed' if tests_passed is not None else 'skipped'} - {test_summary[:80]}"
            )
        return "\n".join(parts)
