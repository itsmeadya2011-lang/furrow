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

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            "After completing the task, return your changes in this format:\n"
            "FILE: path/to/file\n"
            "```\n"
            "<file content>\n"
            "```\n\n"
            "Use one FILE: block per file you modify or create."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._write_files(response)
        return self._summarize_changes(response)

    async def _write_files(self, response: str) -> None:
        """Parse FILE: blocks from the LLM response and write them to disk."""
        pattern = r"FILE:\s*(.+?)\n```\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        for filepath, content in matches:
            filepath = filepath.strip()
            try:
                await self.client.write_file(filepath, content)
            except Exception:
                pass

    def _summarize_changes(self, response: str) -> str:
        """Extract a summary of changes from the LLM response."""
        pattern = r"FILE:\s*(.+?)\n"
        files = re.findall(pattern, response)
        if files:
            return f"Modified/created files: {', '.join(f.strip() for f in files)}"
        return "Task completed (no file changes detected in response)"