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

    async def _read_context(self) -> str:
        """Read task files for context, skipping any that fail."""
        if not self.task.files:
            return ""
        blocks: list[str] = []
        for f in self.task.files:
            try:
                content = await self.client.read_file_lines(f, max_lines=100)
                blocks.append(f"# Context: {f}\n```\n{content}\n```")
            except Exception as e:
                logger.warning("skipping_file_context", file=f, error=str(e))
        return "\n\n".join(blocks)

    async def run(self) -> str:
        logger.info("worker_start", task_id=self.task.id, task=self.task.description)
        context = await self._read_context()
        context_section = f"\n\n## File Context\n{context}\n" if context else ""
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context_section}"
        logger.info("worker_invoke", task_id=self.task.id)
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info("worker_complete", task_id=self.task.id)
        return result
