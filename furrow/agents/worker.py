from __future__ import annotations

import json
import os
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
        # Read context files if specified
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"FILE NOT FOUND: {file_path}"
            except Exception as e:
                file_contents[file_path] = f"ERROR READING: {e}"

        files_context = ""
        if file_contents:
            files_context = "\n\nFile contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn JSON with:\n- summary: what you did\n- changes: array of {{path, content}} for files to write\n- complete: boolean"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            for change in changes:
                path = change.get("path")
                content = change.get("content", "")
                if path:
                    await self.client.write_file(path, content)
            return data.get("summary", "Task completed")
        except (json.JSONDecodeError, ValueError):
            return f"Completed: {response[:200]}"
