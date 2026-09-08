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
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Build context with relevant files
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        context = ""
        if file_contents:
            context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"
        
        # Use the LLM to generate a plan with file operations
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse the response for file operations
        result = await self._execute_file_operations(response)
        return result

    async def _execute_file_operations(self, response: str) -> str:
        """Parse LLM response for file operations and execute them."""
        try:
            # Try to parse as JSON with file operations
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                results = []
                for file_op in data["files"]:
                    path = file_op.get("path")
                    content = file_op.get("content")
                    operation = file_op.get("operation", "write")
                    
                    if operation == "write" and path and content is not None:
                        await self.client.write_file(path, content)
                        results.append(f"Written: {path}")
                    elif operation == "delete" and path:
                        p = Path(path)
                        if p.exists():
                            p.unlink()
                            results.append(f"Deleted: {path}")
                        else:
                            results.append(f"Not found: {path}")
                return "\n".join(results) if results else response
        except json.JSONDecodeError:
            pass
        
        # If not JSON, return the response as-is
        return response