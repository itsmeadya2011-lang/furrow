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
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, workspace: Path | None = None) -> Plan:
        file_list = ""
        if workspace and workspace.exists():
            files = self.client.list_files(workspace)
            if files:
                display_files = files[:50]
                file_list = (
                    f"\nProject files (showing {len(display_files)} of {len(files)}):\n"
                    + "\n".join(display_files)
                )
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{file_list}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
