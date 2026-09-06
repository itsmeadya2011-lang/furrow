from __future__ import annotations

from pathlib import Path
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
        goal: str = "",
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.goal = goal
        self.workspace = workspace or Path.cwd()

    async def _read_context_files(self) -> str:
        if not self.task.files:
            return ""
        context_parts = []
        for file_path in self.task.files[:5]:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"--- {file_path} ---\n{content[:2000]}")
                except Exception:
                    context_parts.append(f"--- {file_path} ---\n[Could not read file]")
        return "\n\n".join(context_parts)

    async def run(self) -> str:
        context = await self._read_context_files()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Goal: {self.goal}\n\n"
            f"Task: {self.task.description}\n\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
        )
        if context:
            prompt += f"Context from existing files:\n{context}\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
