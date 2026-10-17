from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FILE_BLOCK_RE = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written: list[str] = []
        for match in _FILE_BLOCK_RE.finditer(response):
            path = match.group(1).strip()
            start = match.end()
            next_match = _FILE_BLOCK_RE.search(response, start)
            content = response[start:next_match.start()] if next_match else response[start:]
            content = content.strip("\n")
            await self.client.write_file(path, content)
            written.append(path)

        if written:
            return f"Wrote {len(written)} file(s): {', '.join(written)}"
        return "No files were written."