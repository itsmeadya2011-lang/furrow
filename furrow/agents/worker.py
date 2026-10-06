from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        logger.info("Worker starting task", task_id=self.task.id)
        try:
            result = await self.client.complete(
                prompt, model=self.client.settings.worker_model
            )
            logger.info("Worker completed task", task_id=self.task.id)
            return result
        except Exception as e:
            logger.error("Worker task failed", task_id=self.task.id, error=str(e))
            raise