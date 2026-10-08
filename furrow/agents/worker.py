from __future__ import annotations

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
        files = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
            f"Dependencies (already done by other workers): {', '.join(self.task.dependencies) or 'none'}\n\n"
            "Use the available tools (read_file, write_file) to inspect and edit files. "
            "Implement the task completely. When done, reply with a concise summary of what you changed."
        )
        result = await self.client.complete_with_tools(
            prompt,
            system="You are a worker agent implementing a single coding task.",
            model=self.client.settings.worker_model,
        )
        return result or "Task completed."