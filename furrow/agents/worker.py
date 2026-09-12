from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_FILE_BLOCK_RE = re.compile(
    r"```FILE:\s*(?P<path>[^\n]+)\n(?P<content>.*?)```",
    re.DOTALL,
)
_SUMMARY_RE = re.compile(r"SUMMARY:\s*(.+)", re.DOTALL)


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
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._apply_changes(response)
        summary = self._extract_summary(response)
        return summary or "No summary provided."

    async def _apply_changes(self, response: str) -> None:
        for match in _FILE_BLOCK_RE.finditer(response):
            path = match.group("path").strip()
            content = match.group("content")
            await self.client.write_file(path, content)

    @staticmethod
    def _extract_summary(response: str) -> str | None:
        match = _SUMMARY_RE.search(response)
        if match:
            return match.group(1).strip()
        return None