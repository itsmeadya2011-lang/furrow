from __future__ import annotations

import json
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
        await self._apply_changes(response)
        return response

    async def _apply_changes(self, response: str) -> None:
        try:
            data = json.loads(response)
            files = data.get("files", {})
            if isinstance(files, dict):
                for path, content in files.items():
                    await self.client.write_file(path, content)
                return
        except (json.JSONDecodeError, ValueError):
            pass
        if self.task.files:
            await self.client.write_file(self.task.files[0], response)
