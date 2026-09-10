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
        # Read files specified in the task
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        # Build context from files
        context = ""
        if file_contents:
            context = "\n\nFiles:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse JSON response for file operations
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                # Execute file operations
                for file_op in data.get("files", []):
                    path = file_op.get("path")
                    content = file_op.get("content", "")
                    if path:
                        await self.client.write_file(path, content)
                return data.get("summary", "Files updated")
        except (json.JSONDecodeError, ValueError):
            # Not a JSON response, return as-is
            pass
        
        return response
