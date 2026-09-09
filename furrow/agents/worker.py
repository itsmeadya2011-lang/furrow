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
        # Provide file context if files are specified
        file_context = ""
        if self.task.files:
            for f in self.task.files:
                try:
                    content = await self.client.read_file(f)
                    file_context += f"\n--- {f} ---\n{content}\n"
                except FileNotFoundError:
                    file_context += f"\n--- {f} ---\n[File not found, will be created]\n"
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{file_context}\n\nReturn JSON with your changes:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file.py\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"Brief summary of changes\"\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            return data.get("summary", "Task completed")
        except (json.JSONDecodeError, ValueError):
            # Fallback: just return the response as summary
            return response
