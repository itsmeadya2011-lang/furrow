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
        self.workspace = self.client.settings.workspace

    async def run(self) -> str:
        # Read relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)
            else:
                file_contents[file_path] = f"[File not found: {file_path}]"

        files_context = ""
        if file_contents:
            files_context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nWhen you need to read or write files, use the provided tools. Return a concise summary of what you changed."

        # The LLM will respond with a JSON indicating what files to read/write
        # We'll parse this and execute the operations
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        return await self._execute_changes(response)

    async def _execute_changes(self, response: str) -> str:
        """Parse LLM response for file operations and execute them."""
        try:
            # Try to parse as JSON for structured file operations
            data = json.loads(response)
            if isinstance(data, dict) and "operations" in data:
                return await self._apply_operations(data["operations"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # If not structured, return the response as-is (LLM did the work in its mind)
        return response

    async def _apply_operations(self, operations: list[dict]) -> str:
        """Apply a list of file operations."""
        results = []
        for op in operations:
            op_type = op.get("type")
            path = op.get("path")
            if not path:
                continue
            
            full_path = self.workspace / path
            
            if op_type == "write":
                content = op.get("content", "")
                await self.client.write_file(full_path, content)
                results.append(f"Wrote {path}")
            elif op_type == "read":
                content = await self.client.read_file(full_path)
                results.append(f"Read {path}: {content[:100]}...")
            elif op_type == "delete":
                if full_path.exists():
                    full_path.unlink()
                    results.append(f"Deleted {path}")
            elif op_type == "list":
                files = self.client.list_files(full_path)
                results.append(f"Listed {path}: {files}")
        
        return "; ".join(results)
