from __future__ import annotations

import json
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
        # Read existing files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""  # File doesn't exist yet

        # Build context from existing files
        context = ""
        if file_contents:
            context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply the changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                action = change.get("action", "write")
                
                if file_path:
                    await self.client.write_file(file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat response as summary if not valid JSON
            return f"Completed task: {response[:200]}"