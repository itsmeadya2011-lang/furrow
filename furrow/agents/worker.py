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
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Return your response as JSON with this exact shape:
{{
  "changes": [
    {{"file": "relative/path/to/file", "content": "<full file content>"}}
  ],
  "summary": "Concise summary of what you changed"
}}

If no file changes are needed, return {{"changes": [], "summary": "..."}}.
"""
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path is not None and content is not None:
                    await self.client.write_file(file_path, content)

            return summary if summary else "Completed successfully"
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return response