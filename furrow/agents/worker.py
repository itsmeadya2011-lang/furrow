from __future__ import annotations

import json
import os
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
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.client.settings.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        # Build context from files
        context = ""
        if file_contents:
            context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}{context}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        
        # Get the implementation plan from LLM
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse JSON response with file operations
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                # Execute file operations
                results = []
                for file_op in data["files"]:
                    path = file_op.get("path", "")
                    content = file_op.get("content", "")
                    operation = file_op.get("operation", "write")
                    
                    if operation == "write":
                        await self.client.write_file(self.client.settings.workspace / path, content)
                        results.append(f"Created/updated {path}")
                    elif operation == "delete":
                        full_path = self.client.settings.workspace / path
                        if full_path.exists():
                            full_path.unlink()
                            results.append(f"Deleted {path}")
                
                return "; ".join(results) if results else "No file operations performed"
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the raw response as summary
        return response
