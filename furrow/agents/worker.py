from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=config)
        self.config = config or Settings()

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.config.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        # Build prompt with file context
        context = ""
        if file_contents:
            context += "\n\nExisting file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"
        
        # Get implementation from LLM
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse response for file operations
        # Expected format: JSON with file operations
        try:
            import json
            operations = json.loads(response)
            if isinstance(operations, dict) and "files" in operations:
                return await self._execute_file_operations(operations["files"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the raw response as summary
        return response

    async def _execute_file_operations(self, files: list[dict]) -> str:
        """Execute file write operations from the LLM response."""
        results = []
        for file_op in files:
            path = file_op.get("path", "")
            content = file_op.get("content", "")
            operation = file_op.get("operation", "write")
            
            full_path = self.config.workspace / path
            
            if operation == "write" or operation == "create":
                full_path.parent.mkdir(parents=True, exist_ok=True)
                await self.client.write_file(full_path, content)
                results.append(f"Created/updated: {path}")
            elif operation == "delete":
                if full_path.exists():
                    full_path.unlink()
                    results.append(f"Deleted: {path}")
                else:
                    results.append(f"File not found: {path}")
            elif operation == "read":
                if full_path.exists():
                    content = await self.client.read_file(full_path)
                    results.append(f"Read {path}: {content[:200]}...")
                else:
                    results.append(f"File not found: {path}")
        
        return "\n".join(results) if results else "No file operations performed"