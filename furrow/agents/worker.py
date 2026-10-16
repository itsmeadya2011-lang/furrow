from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileChange, TaskModel, WorkerResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> WorkerResult:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            files = [FileChange(**f) for f in data.get("files", [])]
            return WorkerResult(files=files, summary=data.get("summary", "No summary provided"))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return WorkerResult(
                files=[],
                summary=f"Failed to parse worker response: {e}\nRaw response: {response[:500]}",
            )
