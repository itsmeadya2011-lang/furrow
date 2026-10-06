from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read file context for files mentioned in the task
        file_contexts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contexts.append(f"=== {file_path} ===\n{content}")
            except FileNotFoundError:
                file_contexts.append(f"=== {file_path} ===\n(File not found - will be created)")

        file_context = "\n\n".join(file_contexts) if file_contexts else "No existing files to reference."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting file contents:\n{file_context}\n"
        logger.debug("worker_started", task_id=self.task.id, files=self.task.files)
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info("worker_completed", task_id=self.task.id)
        return result
