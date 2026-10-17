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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        existing: list[str] = []
        for path in self.task.files or []:
            if os.path.exists(path):
                try:
                    content = await self.client.read_file(path)
                except Exception:
                    continue
                existing.append(f"### {path}\n```\n{content}\n```")
        if existing:
            prompt += "\n## Existing file contents\n" + "\n".join(existing) + "\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
