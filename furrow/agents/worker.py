from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import TOOL_SPECS, LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self, max_iterations: int = 10) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"Task: {self.task.description}\nFiles to touch: {files_str}\n"
        logger.info(
            "worker_start",
            task_id=self.task.id,
            description=self.task.description,
            files=self.task.files,
            available_tools=[s["name"] for s in TOOL_SPECS],
            max_iterations=max_iterations,
        )
        try:
            summary = await self.client.complete_with_tools(
                prompt=prompt,
                system=WORKER_PROMPT,
                model=self.client.settings.worker_model,
                max_iterations=max_iterations,
            )
            logger.info("worker_complete", task_id=self.task.id, summary_length=len(summary))
            return summary
        except Exception as e:
            logger.exception(
                "worker_failure",
                task_id=self.task.id,
                error=str(e),
            )
            raise
