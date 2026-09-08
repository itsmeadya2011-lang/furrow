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
        self.settings = self.client.settings

    async def run(self) -> str:
        workspace = self.settings.workspace
        file_context = await self._build_file_context(workspace, self.task.files)
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"Workspace files context:\n{file_context}\n"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)

    async def _build_file_context(self, workspace: Path, files: list[str]) -> str:
        if not files:
            return "No specific files mentioned. Explore the workspace as needed."
        parts = []
        for rel in files:
            target = workspace / rel
            if target.exists() and target.is_file():
                try:
                    content = await self.client.read_file(target)
                    parts.append(f"--- {rel} ---\n{content}\n")
                except Exception as e:
                    parts.append(f"--- {rel} (error reading: {e}) ---\n")
            else:
                parts.append(f"--- {rel} (does not exist yet) ---\n")
        return "\n".join(parts) if parts else "No existing files found in context."
