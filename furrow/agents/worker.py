from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        log.info("Worker starting", task_id=self.task.id, description=self.task.description)

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        await self._apply_changes(response)

        log.info("Worker completed", task_id=self.task.id)
        return response

    async def _apply_changes(self, response: str) -> None:
        """Parse file changes from the LLM response and write them to disk."""
        pattern = r"```(?:\w+)?\s*\n?(?:FILE:\s*([^\n]+)\n)?(.*?)```"
        matches = list(re.finditer(pattern, response, re.DOTALL))

        if not matches:
            log.warning("No code blocks found in worker response", task_id=self.task.id)
            return

        for match in matches:
            file_path = match.group(1)
            content = match.group(2).strip()
            if not file_path:
                continue
            await self.client.write_file(file_path, content)
            log.info("Wrote file", task_id=self.task.id, file=file_path)
