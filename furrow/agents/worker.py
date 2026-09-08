from __future__ import annotations

from pathlib import Path
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
        workspace = self.client.settings.workspace
        file_context = ""

        if self.task.files:
            sections = []
            for file_path in self.task.files:
                full_path = Path(workspace) / file_path
                try:
                    content = await self.client.read_file(full_path)
                    sections.append(f"--- {file_path} ---\n{content}")
                except FileNotFoundError:
                    sections.append(f"--- {file_path} ---\n[file does not exist yet]")
                except Exception as e:
                    sections.append(f"--- {file_path} ---\n[could not read: {e}]")
            file_context = "\n\nExisting file contents:\n" + "\n\n".join(sections)
        else:
            files = self.client.list_files()
            tree = "\n".join(files) if files else "(empty workspace)"
            file_context = f"\n\nWorkspace file tree:\n{tree}"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
            f"{file_context}\n"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
