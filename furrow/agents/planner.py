from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.workspace = Path(workspace) if workspace else (settings.workspace if settings else None)

    async def plan(self, goal: str) -> Plan:
        workspace_info = ""
        if self.workspace:
            workspace_info = f"\nWorkspace directory: {self.workspace}\n"
            try:
                files = self.client.list_files(self.workspace)
                if files:
                    workspace_info += f"Existing files (top-level): {', '.join(files[:20])}\n"
            except Exception:
                pass

        prompt = f"{PLANNER_PROMPT}{workspace_info}\n\nGoal: {goal}\n"
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.complete(prompt, model=self.client.settings.planner_model)
                data = json.loads(response)
                return Plan(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Plan parse attempt %s/%s failed: %s", attempt, max_retries, e)
                if attempt == max_retries:
                    raise ValueError(
                        f"Failed to parse plan from LLM after {max_retries} attempts: {e}\nResponse: {response}"
                    ) from e
                prompt = f"{PLANNER_PROMPT}{workspace_info}\n\nGoal: {goal}\n\nPrevious attempt failed to parse. Please return valid JSON only.\n"
