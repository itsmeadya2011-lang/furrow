from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_context = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_context}\n\n"
            f"Before writing code, first read the existing files you need to modify using the "
            f"read_file tool. Then implement the changes using write_file. "
            f"Return a concise summary of what you changed."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._apply_changes(response)
        return response

    async def _apply_changes(self, response: str) -> None:
        """Parse the LLM response for file change blocks and apply them.

        Supports fenced code blocks with file paths, e.g.:

        ```python
        # path/to/file.py
        <content>
        ```
        """
        # Pattern: ```lang\n# path/to/file.py\n<content>```
        pattern = re.compile(
            r"```(\w+)\s*\n#\s*(?P<path>[^\n]+)\n(?P<content>.*?)```",
            re.DOTALL,
        )
        for match in pattern.finditer(response):
            path = match.group("path").strip()
            content = match.group("content")
            try:
                await self.client.write_file(path, content)
            except Exception:
                # Best-effort: don't fail the whole task if one file write fails
                pass