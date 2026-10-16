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
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(path)
            except Exception:
                file_contents[path] = ""

        files_to_touch = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_to_touch}\n"
        )
        if file_contents:
            prompt += "Current file contents:\n"
            for path, content in file_contents.items():
                prompt += f"--- {path} ---\n{content}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            result = json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(response) from exc

        for path, content in result.get("files", {}).items():
            await self.client.write_file(path, content)

        return result["summary"]
