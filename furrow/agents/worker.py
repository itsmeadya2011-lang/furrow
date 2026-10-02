from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.tools import Tool, get_default_tools

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self.task = task
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.tools = tools or get_default_tools()

    async def run(self) -> str:
        files_hint = ", ".join(self.task.files) if self.task.files else "any relevant"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_hint}\n"
            f"Workspace: {self.settings.workspace if self.settings else '.'}\n"
        )
        return await self.client.complete_with_tools(
            prompt,
            model=self.client.settings.worker_model,
            tools=self.tools,
        )
