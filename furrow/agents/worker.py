from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = self.client.settings.workspace

    async def run(self) -> str:
        workspace_info = f"Workspace root: {self.workspace}\n"
        files_info = f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        prompt = f"{WORKER_PROMPT}\n{workspace_info}{files_info}\nTask: {self.task.description}\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
