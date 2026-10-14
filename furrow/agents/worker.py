from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        from furrow.config import settings as global_settings

        self.task = task
        self.client = client or LLMClient(settings=settings or global_settings)

    async def run(self) -> str:
        # Read relevant files
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(self.client.settings.workspace / file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except (OSError, RuntimeError) as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        files_context = (
            "\n\n".join(f"=== {path} ===\n{content}" for path, content in file_contents.items())
            if file_contents
            else "No files specified."
        )

        prompt = f'{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles to modify:\n{files_context}\n\nReturn your response as JSON with this shape:\n{{\n  "changes": [\n    {{\n      "file": "path/to/file.py",\n      "content": "new file content"\n    }}\n  ],\n  "summary": "Brief summary of changes made"\n}}\n'
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse JSON response and apply changes
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(
                        self.client.settings.workspace / file_path, content
                    )

            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: return the raw response if JSON parsing fails
            return f"Task completed (raw response): {response[:500]}"
