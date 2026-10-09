from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


FILE_BLOCK_RE = re.compile(
    r"```file:(?P<path>[^\n]+)\n(?P<content>.*?)```",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        write_file: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self._write_file = write_file or self.client.write_file

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._write_files(response)
        return self._extract_summary(response)

    async def _write_files(self, response: str) -> None:
        for match in FILE_BLOCK_RE.finditer(response):
            file_path = match.group("path").strip()
            content = match.group("content")
            try:
                result = self._write_file(file_path, content)
                if asyncio.iscoroutine(result):
                    await result
            except OSError as e:
                raise RuntimeError(f"Failed to write {file_path}: {e}") from e

    @staticmethod
    def parse_file_blocks(response: str) -> list[tuple[str, str]]:
        return [(m.group("path").strip(), m.group("content")) for m in FILE_BLOCK_RE.finditer(response)]

    @staticmethod
    def _extract_summary(response: str) -> str:
        match = re.search(r"SUMMARY:\s*(.+)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()[:500]
