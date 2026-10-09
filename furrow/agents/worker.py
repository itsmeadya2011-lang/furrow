from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._process_response(response)

    async def _process_response(self, response: str) -> str:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return f"Worker completed task (non-JSON response): {response[:200]}"

        files = data.get("files", [])
        summary = data.get("summary", "Task completed")

        for file_op in files:
            path = file_op.get("path")
            content = file_op.get("content", "")
            if path:
                await self.client.write_file(path, content)

        return summary
