from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings, TaskModel


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(
        self,
        goal: str,
        workspace: Path | None = None,
        existing_tasks: list[TaskModel] | None = None,
    ) -> Plan:
        workspace = workspace or Path.cwd()
        existing_tasks = existing_tasks or []

        files = self.client.list_files(workspace)
        files_str = "\n".join(f"  {f}" for f in files[:100]) if files else "  (empty)"

        completed = [t for t in existing_tasks if t.status == "completed"]
        failed = [t for t in existing_tasks if t.status == "failed"]

        context = f"""Project files (first 100):
{files_str}

Completed tasks:
{json.dumps([{"id": t.id, "description": t.description} for t in completed], indent=2) if completed else "  none"}

Failed tasks:
{json.dumps([{"id": t.id, "description": t.description, "error": t.result} for t in failed], indent=2) if failed else "  none"}"""

        prompt = f"{PLANNER_PROMPT}\n\nContext:\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")