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
        # Read existing files if specified
        file_context = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_context.append(f"=== {file_path} ===\n{content}\n")
            except Exception:
                file_context.append(f"=== {file_path} ===\n[File not found]\n")

        context_str = ""
        if file_context:
            context_str = "\n\nExisting file contents:\n" + "\n".join(file_context)

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context_str}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response.strip())
            if isinstance(data, dict) and "files" in data:
                for file_info in data["files"]:
                    path = file_info.get("path")
                    content = file_info.get("content")
                    if path is not None and content is not None:
                        await self.client.write_file(path, content)
                return data.get("summary", "Done")
        except (json.JSONDecodeError, Exception):
            pass

        return response
