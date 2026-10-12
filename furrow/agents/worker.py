from __future__ import annotations

import json
from pathlib import Path
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
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""

        files_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        ) or "No files specified or found."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles to modify:\n{files_context}\n\nReturn JSON with this shape:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"full file content after changes\"\n    }}\n  ],\n  \"summary\": \"Brief summary of changes made\"\n}}"
        
        response = await self.client.complete(prompt, model=self.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat response as summary
            return f"Completed (parse error: {e}): {response[:200]}"
