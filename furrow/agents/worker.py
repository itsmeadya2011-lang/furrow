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
        self.workspace = Path.cwd()

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = await self.client.read_file(full_path)
                except Exception:
                    file_contents[file_path] = f"[Error reading file: {file_path}]"

        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse the response and apply changes
        try:
            result = json.loads(response)
            changes = result.get("changes", [])
            summary = result.get("summary", "Task completed")
            
            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.write_file(file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: return the raw response
            return response

    async def read_file(self, path: str | Path) -> str:
        return await self.client.read_file(self.workspace / path)

    async def write_file(self, path: str | Path, content: str) -> None:
        await self.client.write_file(self.workspace / path, content)

    def list_files(self, directory: str | Path = ".") -> list[str]:
        return self.client.list_files(self.workspace / directory)
