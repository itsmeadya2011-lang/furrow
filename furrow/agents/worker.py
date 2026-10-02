from __future__ import annotations

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

    async def _gather_file_context(self) -> str:
        """Read the contents of files the task references for LLM context."""
        if not self.task.files:
            return ""
        context_parts: list[str] = []
        for rel_path in self.task.files:
            path = self.client.settings.workspace / rel_path
            if path.exists():
                content = await self.client.read_file(path)
                context_parts.append(f"--- {rel_path} ---\n{content}")
        return "\n\n".join(context_parts)

    async def run(self) -> str:
        file_context = await self._gather_file_context()
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
        )
        if file_context:
            prompt += f"\nExisting file context:\n{file_context}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # If the LLM response contains structured file writes, persist them.
        # Format expected: "FILE: <path>" blocks with code fences.
        if "FILE:" in result:
            await self._apply_file_edits(result)

        return result

    async def _apply_file_edits(self, response: str) -> None:
        """Parse and persist file edits from a structured LLM response.

        Supported format:
            FILE: path/to/file.py
            ```python
            ...content...
            ```

            FILE: path/to/other.py
            ```python
            ...
            ```
        """
        pattern = re.compile(
            r"FILE:\s*(.+?)\n```[\w]*\n(.*?)```",
            re.DOTALL,
        )
        for match in pattern.finditer(response):
            file_path = match.group(1).strip()
            content = match.group(2)
            full_path = self.client.settings.workspace / file_path
            await self.client.write_file(full_path, content)
