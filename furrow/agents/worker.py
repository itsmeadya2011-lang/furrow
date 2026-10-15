from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from furrow.config import Settings


_FILE_BLOCK_RE = re.compile(
    r"FILE:\s*(?P<path>[^\n]+)\n\s*```[a-zA-Z]*\n(?P<content>.*?)```",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_start", task_id=self.task.id)
        logger.info("Task execution started", task_id=self.task.id, description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        written = await self._apply_changes(result)
        if written:
            summary = f"Wrote {len(written)} files: {', '.join(written)}"
        else:
            summary = "No files were written."
        logger.info("Task execution completed", task_id=self.task.id, files_written=len(written))
        return summary

    def _parse_file_changes(self, response: str) -> list[tuple[str, str]]:
        """Parse ``FILE:`` code-block blocks from an LLM response.

        Recognised block format::

            FILE: path/to/file.py
            ```language
            <full file content>
            ```

        Tolerates backticks around the path and optional blank lines between
        the ``FILE:`` marker and the opening fence.
        """
        changes: list[tuple[str, str]] = []
        for match in _FILE_BLOCK_RE.finditer(response):
            file_path = match.group("path").strip().strip("`").strip()
            content = match.group("content")
            if file_path:
                changes.append((file_path, content))
        return changes

    async def _apply_changes(self, response: str) -> list[str]:
        """Parse file changes from the response and write them to disk.

        Returns the list of file paths that were written.
        """
        written: list[str] = []
        for file_path, content in self._parse_file_changes(response):
            path = Path(file_path)
            if not path.is_absolute():
                path = self.client.settings.workspace / path
            await self.client.write_file(path, content)
            logger.info("File written", task_id=self.task.id, path=str(path))
            written.append(file_path)
        return written

    def __repr__(self) -> str:
        return f"WorkerAgent(task_id={self.task.id!r})"
