from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        log.info("task_started", task_id=self.task.id, description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        log.info("task_finished", task_id=self.task.id, result_len=len(result))
        return result
