from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or default_settings)

    async def run(self) -> str:
        # First, read any existing files to understand context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""  # New file
            except OSError as e:
                file_contents[file_path] = f"Error reading file: {e}"

        # Build prompt with file context
        files_context = ""
        if file_contents:
            files_context = "\n\nExisting file contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        
        # Get LLM response - it should return JSON with file operations
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse as JSON with file operations
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                # Execute file operations
                for file_op in data["files"]:
                    path = file_op.get("path", "")
                    content = file_op.get("content", "")
                    operation = file_op.get("operation", "write")
                    if path and operation in ("write", "create"):
                        await self.client.write_file(path, content)
                return data.get("summary", "Completed file operations")
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the raw response
        return response
