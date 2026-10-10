from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        """Execute the task using LLM tool-use to read and write files.

        The worker LLM can call read_file, write_file, and list_files to
        implement the task.  The final text response (summary) is returned.
        """
        files_str = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task ID: {self.task.id}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
        )
        logger.info("worker_start", task_id=self.task.id, description=self.task.description)
        result = await self.client.complete_with_tools(
            prompt,
            model=self.client.settings.worker_model,
        )
        logger.info("worker_done", task_id=self.task.id, result_length=len(result))
        return result
