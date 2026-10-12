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
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                file_contents[file_path] = await self.client.read_file(file_path)
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except OSError as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        files_context = (
            "\n\n".join(f"=== {path} ===\n{content}" for path, content in file_contents.items())
            if file_contents
            else "No files to read."
        )

        prompt = f'{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {", ".join(self.task.files) if self.task.files else "any"}\n\nCurrent file contents:\n{files_context}\n\nReturn JSON with your changes:\n{{\n  "changes": [\n    {{\n      "file": "path/to/file.py",\n      "content": "new file content"\n    }}\n  ],\n  "summary": "Brief summary of changes"\n}}'

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)

            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: return the raw response
            return f"Worker completed (raw response): {response[:500]}"