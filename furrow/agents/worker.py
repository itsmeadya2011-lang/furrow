from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_USER_TEMPLATE
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = WORKER_USER_TEMPLATE.format(
            description=self.task.description,
            files=', '.join(self.task.files) if self.task.files else 'any',
        )
        return await self.client.complete(
            prompt, system=WORKER_PROMPT, model=self.client.settings.worker_model
        )
