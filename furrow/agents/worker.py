from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # First, read any existing files to understand context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""

        # Build prompt with file context
        files_context = ""
        if file_contents:
            files_context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn JSON with this shape:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"Brief description of changes\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")
            
            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            
            return f"{summary}\nChanged files: {', '.join([c.get('file', '') for c in changes])}"
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat response as summary
            return f"Worker completed (parse error: {e}): {response[:200]}"
