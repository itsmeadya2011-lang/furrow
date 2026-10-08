from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        log.info("worker_start", task_id=self.task.id, description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        try:
            result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        except Exception as e:
            log.warning("worker_complete_failed", task_id=self.task.id, error=str(e))
            raise
        log.info("worker_complete", task_id=self.task.id, response_length=len(result))
        return result