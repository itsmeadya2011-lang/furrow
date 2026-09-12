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

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        workspace = self.client.settings.workspace
        
        for file_path in self.task.files:
            full_path = workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    file_contents[file_path] = content
                except Exception as e:
                    file_contents[file_path] = f"Error reading file: {e}"
        
        # Also list files in workspace if no specific files given
        if not self.task.files:
            all_files = self.client.list_files(workspace)
            # Limit to first 50 files to avoid token limits
            context_files = all_files[:50]
            for file_path in context_files:
                full_path = workspace / file_path
                if full_path.suffix in ['.py', '.js', '.ts', '.json', '.md', '.txt', '.yaml', '.yml', '.toml']:
                    try:
                        content = await self.client.read_file(full_path)
                        file_contents[file_path] = content[:2000]  # Truncate large files
                    except Exception:
                        pass

        # Build context
        context = ""
        if file_contents:
            context += "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse response as file operations
        return await self._apply_changes(response, file_contents)

    async def _apply_changes(self, response: str, file_contents: dict[str, str]) -> str:
        """Try to parse LLM response as file operations and apply them."""
        try:
            # Try to parse as JSON with file operations
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                results = []
                for file_op in data["files"]:
                    path = file_op.get("path")
                    content = file_op.get("content")
                    operation = file_op.get("operation", "write")
                    
                    if path and content is not None:
                        full_path = self.client.settings.workspace / path
                        if operation == "write":
                            await self.client.write_file(full_path, content)
                            results.append(f"Written: {path}")
                        elif operation == "append":
                            existing = file_contents.get(path, "")
                            await self.client.write_file(full_path, existing + "\n" + content)
                            results.append(f"Appended: {path}")
                        elif operation == "delete":
                            full_path.unlink(missing_ok=True)
                            results.append(f"Deleted: {path}")
                return "; ".join(results) if results else "No file operations performed"
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the raw response as summary
        return response[:500] + ("..." if len(response) > 500 else "")
