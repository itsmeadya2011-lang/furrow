from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_start", task_id=self.task.id, description=self.task.description[:80])
        self.task.mark_in_progress()

        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_str}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        self.task.mark_completed(result)
        logger.info("worker_done", task_id=self.task.id)
        return result
