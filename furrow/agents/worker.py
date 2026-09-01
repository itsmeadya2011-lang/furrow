from __future__ import annotations

import json
import os
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
        # Read files for context
        file_context = ""
        for f in self.task.files:
            try:
                content = await self.client.read_file(f)
                file_context += f"\n--- {f} ---\n{content}\n"
            except Exception as e:
                file_context += f"\n--- {f} --- (could not read: {e})\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if file_context:
            prompt += f"\nExisting file context:\n{file_context}\n"

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        # Try to parse structured JSON response and write files
        try:
            data = json.loads(response)
            summary = data.get("summary", "Task completed.")
            files = data.get("files", {})
            for file_path, file_content in files.items():
                await self.client.write_file(file_path, file_content)
            if files:
                summary += f"\nWrote {len(files)} file(s)."
            return summary
        except (json.JSONDecodeError, ValueError):
            # Fall back to raw text response
            return response
