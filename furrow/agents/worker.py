from __future__ import annotations

from pathlib import Path
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
        file_context = await self._read_files()
        prompt = self._build_prompt(file_context)
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info(
            "worker_completed",
            task_id=self.task.id,
            files_read=len(file_context),
        )
        return result

    async def _read_files(self) -> dict[str, str]:
        """Read files mentioned in the task to provide context to the LLM."""
        contents: dict[str, str] = {}
        for f in self.task.files:
            try:
                content = await self.client.read_file(f)
                contents[f] = content
            except (FileNotFoundError, OSError) as e:
                logger.warning("worker_read_file_failed", path=f, error=str(e))
        return contents

    def _build_prompt(self, file_context: dict[str, str]) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
        if self.task.files:
            prompt += f"Files to touch: {', '.join(self.task.files)}\n"
        if self.task.dependencies:
            prompt += f"Depends on: {', '.join(self.task.dependencies)}\n"
        if file_context:
            prompt += "\nCurrent file contents:\n"
            for path, content in file_context.items():
                prompt += f"\n--- {path} ---\n{content}\n"
        return prompt