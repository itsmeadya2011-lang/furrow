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
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        from furrow.config import settings as default_settings

        self.task = task
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> str:
        files_hint = (
            ", ".join(self.task.files) if self.task.files else "any relevant files"
        )
        deps_hint = (
            f"\nDependencies: {', '.join(self.task.dependencies)}"
            if self.task.dependencies
            else ""
        )
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_hint}"
            f"{deps_hint}\n"
        )
        return await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )
