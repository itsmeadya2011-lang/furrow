from __future__ import annotations

import re
from pathlib import Path
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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._write_files(response)
        return response

    async def _write_files(self, response: str) -> None:
        pattern = re.compile(r"FILE:\s*(.+?)\n(.*?)(?=\nFILE:|\Z)", re.DOTALL)
        for match in pattern.finditer(response):
            file_path = match.group(1).strip()
            content = match.group(2).strip()
            if file_path and content:
                try:
                    await self.client.write_file(Path(file_path), content)
                except Exception:
                    pass
