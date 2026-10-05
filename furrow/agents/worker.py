from __future__ import annotations

from furrow.agents.base import BaseAgent, SupportsComplete
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel


class WorkerAgent(BaseAgent):
    """Implements a single task and returns the worker's raw report text."""

    model_attr = "worker_model"

    def __init__(
        self,
        task: TaskModel,
        client: SupportsComplete | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(client=client, settings=settings)
        self.task = task

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"Task: {self.task.description}\nFiles to touch: {files}\n"
        return await self.ask(prompt, system=WORKER_PROMPT)