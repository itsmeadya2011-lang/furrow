from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import FILE_TOOLS, LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        model = getattr(getattr(self.client, "settings", None), "worker_model", None)
        return await self.client.run_tool_loop(
            prompt=prompt,
            system=WORKER_PROMPT,
            model=model,
            tools=FILE_TOOLS,
            max_rounds=10,
        )
