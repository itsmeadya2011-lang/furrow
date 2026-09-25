from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Build context with file contents if files are specified
        file_context = ""
        if self.task.files:
            file_contents = []
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    file_contents.append(f"=== {file_path} ===\n{content}")
                except FileNotFoundError:
                    file_contents.append(f"=== {file_path} === (file not found, will be created)")
            file_context = "\n\nRelevant files:\n" + "\n\n".join(file_contents)

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{file_context}\n"
        )

        model = self.settings.get_model_for_agent("worker")
        return await self.client.complete(prompt, model=model)

    async def read_file(self, path: str) -> str:
        """Allow worker to read files."""
        return await self.client.read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        """Allow worker to write files."""
        await self.client.write_file(path, content)

    def list_files(self, directory: str = ".") -> list[str]:
        """Allow worker to list files."""
        return self.client.list_files(directory)