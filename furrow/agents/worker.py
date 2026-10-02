from __future__ import annotations

import os
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

    async def run(self) -> str:
        context = ""
        if self.task.files:
            context_lines = []
            for file_path in self.task.files:
                full_path = os.path.join(self.client.settings.workspace, file_path)
                try:
                    content = await self.client.read_file(full_path)
                except (FileNotFoundError, IsADirectoryError, PermissionError):
                    content = "<file does not exist yet>"
                context_lines.append(f"--- {file_path} ---\n{content}")
            context = "=== Existing file context ===\n" + "\n\n".join(context_lines) + "\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{context}"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
