from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.planner import _strip_fences
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
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(_strip_fences(response))
            files_written = data.get("files_written", [])
            for entry in files_written:
                path = Path(entry["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(entry["content"])
            summary = data.get("summary", response)
            return summary
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            return response
