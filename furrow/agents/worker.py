from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
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

    async def run(self) -> str:
        logger.info(
            "worker_task_start",
            task_id=self.task.id,
            description=self.task.description,
        )
        files_context = await self._build_files_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"\nFile contents provided for context:\n{files_context}"
        )
        result = await self.client.complete(
            prompt,
            model=self.client.settings.worker_model,
        )
        logger.info("worker_task_complete", task_id=self.task.id)
        return result

    async def _build_files_context(self) -> str:
        sections: list[str] = []
        for path in self.task.files:
            logger.debug("reading_file", path=path)
            try:
                content = await self.client.read_file(path)
                sections.append(f"--- File: {path} ---\n{content}\n")
            except FileNotFoundError:
                logger.warning("file_not_found", path=path)
                sections.append(f"--- File: {path} ---\n[File does not exist]\n")
        return "\n".join(sections)
