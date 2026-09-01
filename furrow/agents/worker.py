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
        # Read existing files if specified
        file_contents = {}
        for file_path in self.task.files:
            try:
                file_contents[file_path] = await self.client.read_file(file_path)
            except (FileNotFoundError, OSError):
                file_contents[file_path] = ""

        # Build prompt with file context
        files_context = ""
        for path, content in file_contents.items():
            files_context += f"\n--- {path} ---\n{content}\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{files_context}\n"
            "Return JSON with this shape:\n"
            "{\n"
            "  'changes': [{'path': 'file.py', 'content': 'full file content'}],\n"
            "  'summary': 'Brief description of changes made'\n"
            "}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the response and apply changes
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Changes applied")

            for change in changes:
                path = change.get("path", "")
                content = change.get("content", "")
                if path and content:
                    await self.client.write_file(path, content)

            return summary
        except (json.JSONDecodeError, ValueError):
            # If LLM didn't return valid JSON, just return the raw response
            return response
