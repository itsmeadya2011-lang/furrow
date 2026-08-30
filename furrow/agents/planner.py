from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


EXCLUDED_PATTERNS = (
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    ".eggs",
)


def _is_excluded(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDED_PATTERNS:
            return True
        if part.endswith(".egg-info"):
            return True
    return False


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        files = await self.client.list_files(self.client.settings.workspace)
        filtered = [f for f in files if not _is_excluded(f)]
        file_tree = "\n".join(f"  {f}" for f in sorted(filtered))

        prompt = (
            f"{PLANNER_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Project file tree:\n{file_tree}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
