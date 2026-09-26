from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"Task: {self.task.description}\nFiles to touch: {files_str}\n"
        return await self.client.complete(
            prompt=prompt,
            system=WORKER_PROMPT,
            model=self.client.settings.worker_model,
        )
