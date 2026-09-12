from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.tool_loop import run_tool_loop
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        return await run_tool_loop(
            client=self.client,
            system_prompt=WORKER_PROMPT,
            user_message=(
                f"Task: {self.task.description}\n"
                f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            ),
            max_rounds=10,
            workspace=str(self.client.settings.workspace),
            model=self.client.settings.worker_model,
        )
