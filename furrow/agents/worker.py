from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        cycle: int = 1,
        goal: str = "",
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.cycle = cycle
        self.goal = goal
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = WORKER_PROMPT.format(
            goal=self.goal,
            cycle=self.cycle,
        ) + f"\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
