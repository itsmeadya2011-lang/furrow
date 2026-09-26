from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import get_logger

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    """Executes a single assigned task from a Plan."""

    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.debug("worker_result", task_id=self.task.id, result_len=len(result))
        return result
