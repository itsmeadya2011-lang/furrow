from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> dict[str, Any]:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                for file_path, content in data["files"].items():
                    await self.client.write_file(file_path, content)
                return {"status": "completed", "files_written": list(data["files"].keys()), "summary": data.get("summary", response)}
        except (json.JSONDecodeError, ValueError):
            pass
        return {"status": "completed", "files_written": [], "summary": response}
