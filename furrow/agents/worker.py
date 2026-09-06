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
        tools = self.client.get_file_tools()
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"Implement the following task. You have access to file tools.\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
            f"Workspace: {self.client.settings.workspace}\n\n"
            f"Use the tools to read, modify, and write files. After completing the task, "
            f"provide a concise summary of what was changed."
        )
        return await self.client.complete(
            prompt,
            system=WORKER_PROMPT,
            model=self.client.settings.worker_model,
            tools=tools,
        )
