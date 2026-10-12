from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


logger = get_logger("planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path.cwd()

    async def plan(self, goal: str) -> Plan:
        logger.info("planning_started", goal=goal)
        # First, explore the codebase to provide context
        context = await self._gather_context()

        prompt = f"{PLANNER_PROMPT}\n\nCodebase Context:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("planning_completed", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("planning_failed", error=str(e), response=response[:500])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _gather_context(self) -> str:
        """Gather relevant codebase context for planning."""
        parts = []

        # List all files
        all_files = self.client.list_files(self.workspace)
        if all_files:
            parts.append(f"Project files ({len(all_files)} total):\n" + "\n".join(sorted(all_files)))

        # Read key configuration files
        key_files = [
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "requirements.txt",
            "setup.py",
            "README.md",
            "README.rst",
        ]
        for key_file in key_files:
            path = self.workspace / key_file
            if path.exists():
                try:
                    content = await self.client.read_file(path)
                    parts.append(f"\n--- {key_file} ---\n{content[:2000]}")
                except Exception:
                    pass

        # Check for test directories
        test_dirs = ["tests", "test", "spec", "__tests__"]
        for test_dir in test_dirs:
            path = self.workspace / test_dir
            if path.exists() and path.is_dir():
                test_files = self.client.list_files(path)
                if test_files:
                    parts.append(f"\n--- Test files in {test_dir} ---\n" + "\n".join(test_files[:20]))

        return "\n".join(parts) if parts else "No project files found (empty directory)."