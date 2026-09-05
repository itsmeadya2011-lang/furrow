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
        files_section = ""
        if self.task.files:
            files_section = "\nRelevant files:\n"
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    preview = content[:3000]
                    files_section += f"\n--- {file_path} ---\n{preview}\n"
                except Exception:
                    files_section += f"\n--- {file_path} ---\n(file not found or unreadable)\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
            f"{files_section}\n"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
