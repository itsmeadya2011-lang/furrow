from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Read relevant files
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.settings.workspace / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = await self.client.read_file(full_path)
                except Exception as e:  # noqa: BLE001
                    file_contents[file_path] = f"[Error reading file: {e}]"

        # Build context
        context = ""
        if file_contents:
            context += "Current file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n{context}\n\nReturn your changes as JSON with this format:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"new file content\"\n    }}\n  ],\n  \"summary\": \"Brief description of changes made\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and apply changes
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    full_path = self.settings.workspace / file_path
                    await self.client.write_file(full_path, content)

            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: return the raw response
            return f"Completed (raw): {response[:200]}"
