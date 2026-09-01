from __future__ import annotations

import re
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

_FILE_MARKER = re.compile(r"^=== FILE:\s*(.+?)\s*===\s*$", re.MULTILINE)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        logger.info("worker.run.started", task_id=self.task.id, files=self.task.files)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info("worker.run.completed", task_id=self.task.id)

        if self.task.files:
            if len(self.task.files) == 1:
                target = self.task.files[0]
                logger.info("worker.write_file.started", task_id=self.task.id, file=target)
                await self.client.write_file(target, response)
                logger.info("worker.write_file.completed", task_id=self.task.id, file=target)
            else:
                # Multi-file mode: parse === FILE: path === markers from the response
                files_written = await self._write_multiple_files(response)
                if files_written == 0:
                    logger.warning(
                        "worker.write_file.skipped",
                        task_id=self.task.id,
                        reason="multiple files specified without structured output",
                    )

        return response

    async def _write_multiple_files(self, response: str) -> int:
        """Parse === FILE: path === markers and write each section to its file.

        Returns the number of files successfully written.
        """
        matches = list(_FILE_MARKER.finditer(response))
        if not matches:
            return 0

        written = 0
        for i, match in enumerate(matches):
            filepath = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
            content = response[start:end].strip()
            logger.info("worker.write_file.started", file=filepath)
            await self.client.write_file(filepath, content)
            logger.info("worker.write_file.completed", file=filepath)
            written += 1
        return written
