from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Executes a single planned task using the LLM and file tools."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        files = ", ".join(self.task.files) if self.task.files else "any"
        system = (
            "You are a meticulous coding agent in an autonomous development loop. "
            "Use the available file tools to read, write and list files in the "
            "workspace. Make minimal, targeted changes that complete the assigned "
            "task. Return a concise summary of what you changed."
        )
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Workspace root: {workspace}\n"
            f"Files to touch: {files}\n\n"
            f"Task (id={self.task.id}): {self.task.description}\n\n"
            f"When you are done, return a concise summary of what you changed."
        )
        return await self.client.complete_with_tools(
            prompt, system=system, model=self.client.settings.worker_model
        )
