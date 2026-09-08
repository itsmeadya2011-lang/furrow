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
        files_to_read = self.task.files or self.client.list_files(self.client.settings.workspace)
        file_contents = {}
        for rel_path in files_to_read:
            abs_path = self.client.settings.workspace / rel_path
            if abs_path.is_file():
                try:
                    file_contents[rel_path] = await self.client.read_file(abs_path)
                except Exception:
                    pass

        file_section = "\n".join(f"--- {path} ---\n{content}" for path, content in file_contents.items())
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting files:\n{file_section}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            files = data.get("files", {})
            for rel_path, content in files.items():
                abs_path = self.client.settings.workspace / rel_path
                await self.client.write_file(abs_path, content)
            summary = data.get("summary", response)
            return summary
        except (json.JSONDecodeError, ValueError):
            return response
