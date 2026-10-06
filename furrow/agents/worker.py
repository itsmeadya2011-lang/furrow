from __future__ import annotations

import json
from pathlib import Path
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
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        file_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        ) if file_contents else "No files specified."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles to modify:\n{file_context}\n\nReturn a JSON object with your changes:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"new file content\"\n    }}\n  ],\n  \"summary\": \"Brief description of changes made\"\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return raw response
            return f"Completed (raw): {response[:200]}"
