from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from furrow.agents.parser import parse_file_edits
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        edits = parse_file_edits(response)
        for path, content in edits:
            logger.debug("Worker writing file: %s", path)
            await self.client.write_file(path, content)

        if not edits:
            return response.strip() or "No changes were needed."

        paths = list(dict.fromkeys(path for path, _ in edits))
        summary = f"Modified {len(paths)} file(s): {', '.join(paths)}"
        logger.info("Task %s: %s", self.task.id, summary)
        return summary
