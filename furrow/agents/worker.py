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
        # Read relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        
        # The LLM should return a JSON with the changes to make
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse as JSON for structured changes
        try:
            changes = json.loads(response)
            if isinstance(changes, dict) and "changes" in changes:
                return await self._apply_changes(changes["changes"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the raw response as summary
        return response

    async def _apply_changes(self, changes: list[dict]) -> str:
        """Apply a list of file changes. Each change has 'path' and 'content'."""
        results = []
        for change in changes:
            path = change.get("path")
            content = change.get("content")
            if path and content is not None:
                await self.client.write_file(path, content)
                results.append(f"Updated {path}")
        return "; ".join(results) if results else "No changes applied"
