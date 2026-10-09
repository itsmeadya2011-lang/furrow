from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self) -> str:
        task = self.task
        logger.info("worker_started", task_id=task.id, description=task.description)

        prompt = f"{WORKER_PROMPT}\n\nTask: {task.description}\nFiles to touch: {', '.join(task.files) if task.files else 'any'}\n"
        try:
            result = await self.client.complete(prompt, model=self.client.settings.worker_model)
            logger.info("worker_completed", task_id=task.id)
            return result
        except Exception as e:
            logger.error("worker_failed", task_id=task.id, error=str(e))
            raise
