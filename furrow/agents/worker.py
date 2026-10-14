from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.context import ContextManager
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        context: ContextManager | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.context = context

    async def run(self) -> str:
        # Build context with file contents if files are specified
        file_context = ""
        if self.task.files:
            file_contents = []
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    file_contents.append(f"=== {file_path} ===\n{content}")
                except FileNotFoundError:
                    file_contents.append(f"=== {file_path} ===\n(File not found - will be created)")
            if file_contents:
                file_context = "\n\nRelevant files:\n" + "\n\n".join(file_contents)

        # Add context from context manager
        context_str = ""
        if self.context and self.task.files:
            context_str = self.context.get_context_for_worker(self.task.files)
            if context_str:
                context_str = f"\n\nContext:\n{context_str}"

        prompt = f'{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {", ".join(self.task.files) if self.task.files else "any"}{file_context}{context_str}\n\nReturn JSON with this shape:\n{{\n  "changes": [\n    {{"file": "path/to/file.py", "content": "new file content"}}\n  ],\n  "summary": "What was changed"\n}}'

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

            return json.dumps({"changes": changes, "summary": summary})
        except (json.JSONDecodeError, ValueError):
            # Fallback: return raw response
            return f"Task completed (raw response): {response[:500]}"
