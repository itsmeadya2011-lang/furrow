from __future__ import annotations

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient) -> None:
        self.task = task
        self.client = client

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
