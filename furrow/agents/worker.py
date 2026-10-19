from __future__ import annotations

import json
import re
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
        # Gather existing file contents for context
        existing_files = {}
        if self.task.files:
            for file_path in self.task.files:
                try:
                    existing_files[file_path] = await self.client.read_file(file_path)
                except Exception:
                    existing_files[file_path] = ""

        files_context = ""
        if existing_files:
            files_context = "\nExisting files:\n"
            for path, content in existing_files.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{files_context}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from a fenced code block
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
            else:
                raise ValueError(f"Worker returned non-JSON response: {response}")

        changes = data.get("changes", [])
        summary = data.get("summary", "No summary provided.")

        for change in changes:
            file_path = change.get("file")
            content = change.get("content", "")
            if not file_path:
                continue
            await self.client.write_file(file_path, content)

        return summary
