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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)

    async def run_with_tools(self) -> str:
        result = await self.run()
        workdir = Path(".workdir")
        workdir.mkdir(parents=True, exist_ok=True)
        artifact = workdir / f"task-{self.task.id}.txt"
        await self.client.write_file(artifact, result)
        return result

    def summarize(self, response: str) -> dict:
        try:
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("response is not a JSON object")
            return {
                "status": str(data.get("status", "unknown")),
                "files_changed": list(data.get("files_changed", [])),
                "summary": str(data.get("summary", "")),
                "tests": str(data.get("tests", "n/a")),
            }
        except (json.JSONDecodeError, ValueError):
            return {
                "status": "unknown",
                "files_changed": [],
                "summary": response,
                "tests": "n/a",
            }
