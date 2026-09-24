from __future__ import annotations

import asyncio
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
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None, workspace: Path | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()

    async def run(self) -> str:
        # Read existing files if specified
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)
        
        file_context = ""
        if file_contents:
            file_context = "\n\nExisting file contents:\n"
            for path, content in file_contents.items():
                file_context += f"\n--- {path} ---\n{content}\n"
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{file_context}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse the JSON response and apply changes
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            complete = data.get("complete", True)
            
            for change in changes:
                path = change.get("path")
                content = change.get("content", "")
                if path:
                    full_path = self.workspace / path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    await self.client.write_file(full_path, content)
            
            if complete:
                self.task.status = "completed"
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat response as plain text summary
            return response
