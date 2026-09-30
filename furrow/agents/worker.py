from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import logger

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Agent that implements a single assigned task."""

    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        log = logger.bind(task_id=self.task.id, task_desc=self.task.description)
        log.debug("worker_started")
        files_hint = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_hint}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        log.info("worker_completed", result_length=len(result))
        return result
