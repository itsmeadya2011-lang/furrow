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
        sections = [WORKER_PROMPT]
        sections.append(f"Task: {self.task.description}")

        file_entries = []
        for file_path in self.task.files or []:
            try:
                content = await self.client.read_file(file_path)
                file_entries.append(f"- {file_path}:\n```\n{content}\n```")
            except Exception as e:
                file_entries.append(f"- {file_path}: (unable to read: {e})")

        if file_entries:
            sections.append("Files to modify:\n" + "\n".join(file_entries))

        try:
            root_files = self.client.list_files(".")
            if root_files:
                sections.append("Project structure:\n" + "\n".join(f"- {f}" for f in root_files))
        except Exception:
            pass

        prompt = "\n\n".join(sections)
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
