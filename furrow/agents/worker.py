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
        # Read existing files for context
        file_context = ""
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_context += f"\n--- {file_path} ---\n{content}\n"
            except FileNotFoundError:
                file_context += f"\n--- {file_path} ---\n(File does not exist)\n"
            except Exception as e:
                file_context += f"\n--- {file_path} ---\n(Error reading: {e})\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting file contents:\n{file_context}\n\nReturn JSON with this shape:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file.py\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"Brief summary of changes\"\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract any file writes from the response
            return f"Completed (raw response): {response[:500]}"
