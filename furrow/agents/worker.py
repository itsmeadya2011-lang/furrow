from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

logger = structlog.get_logger()

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_start", task_id=self.task.id, description=self.task.description[:100])
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info("worker_complete", task_id=self.task.id, result_length=len(result))
        return result
