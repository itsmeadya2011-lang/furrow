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
        # Get existing file contents for context
        file_contents = ""
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents += f"\n--- {file_path} ---\n{content}\n"
            except FileNotFoundError:
                file_contents += f"\n--- {file_path} ---\n(File does not exist yet)\n"
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting file contents:{file_contents}\n\nImplement the task completely. Use the read_file, write_file, and run_command tools as needed."
        
        return await self.client.run_agent(
            prompt=prompt,
            model=self.client.settings.worker_model,
            max_rounds=5
        )
