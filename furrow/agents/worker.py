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

        result = json.loads(response)
        summary = result.get("summary", "")
        files_changed = result.get("files_changed", [])

        for file_path in files_changed:
            try:
                current_content = await self.client.read_file(file_path)
            except FileNotFoundError:
                current_content = ""
            file_prompt = f"Write the full updated content for file {file_path} given the task: {self.task.description}\n\nCurrent content:\n{current_content}\n"
            new_content = await self.client.complete(file_prompt, model=self.client.settings.worker_model)
            await self.client.write_file(file_path, new_content)

        return summary
