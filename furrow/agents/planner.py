from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    def _get_project_context(self) -> str:
        """Gather project context: file listing, git status, etc."""
        parts = []
        workspace = self.settings.workspace

        # File listing
        files = self.client.list_files(workspace)
        if files:
            parts.append("## Project Files\n```\n" + "\n".join(sorted(files)[:100]) + "\n```")
        else:
            parts.append("## Project Files\n(Empty directory)")

        # Git status
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts.append(f"## Git Status\n```\n{result.stdout.strip()}\n```")
            elif result.returncode == 0:
                parts.append("## Git Status\n(Working tree clean)")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Git log (recent commits)
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-5"],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts.append(f"## Recent Commits\n```\n{result.stdout.strip()}\n```")
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return "\n\n".join(parts)

    async def plan(self, goal: str, cycle_history: list[dict] | None = None) -> Plan:
        context = self._get_project_context()

        history_str = ""
        if cycle_history:
            history_str = "\n## Previous Cycles\n"
            for cycle in cycle_history[-3:]:  # Last 3 cycles
                history_str += f"\n### Cycle {cycle['cycle']}\n"
                history_str += f"Plan: {cycle['plan'].get('rationale', 'N/A')}\n"
                for task in cycle.get("tasks", []):
                    history_str += f"  - Task {task['id']}: {task['description'][:60]} ({task['status']})\n"
                if cycle.get("test_result"):
                    tr = cycle["test_result"]
                    history_str += f"  Tests: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['summary']}\n"

        prompt = f"{PLANNER_PROMPT}\n\n{context}\n{history_str}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
