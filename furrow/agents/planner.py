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
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace

    async def plan(self, goal: str) -> Plan:
        workspace_files = ""
        if self.workspace is not None:
            try:
                files = self.client.list_files(self.workspace)
                workspace_files = "\n".join(f"  - {f}" for f in files)
                if not files:
                    workspace_files = "  (empty directory)"
            except Exception:
                workspace_files = "  (could not list files)"

        prompt = (
            f"{PLANNER_PROMPT.format(workspace_files=workspace_files)}\n\n"
            f"Goal: {goal}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
