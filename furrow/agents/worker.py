from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
        file_listing: str | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace
        self.file_listing = file_listing

    async def run(self, workspace: Path | None = None) -> str:
        effective_workspace = workspace or self.workspace
        file_listing = self.file_listing

        if effective_workspace is not None and file_listing is None:
            file_listing = "\n".join(self.client.list_files(str(effective_workspace)))

        task_context = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
        if file_listing:
            task_context += f"\n\nProject structure:\n{file_listing}"

        return await self.client.complete(task_context, system=WORKER_PROMPT, model=self.client.settings.worker_model)
