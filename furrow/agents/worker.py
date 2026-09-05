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
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace

    async def run(self) -> str:
        file_contents = ""
        if self.workspace is not None and self.task.files:
            lines = []
            for filename in self.task.files:
                try:
                    content = await self.client.read_file(self.workspace / filename)
                    lines.append(f"### {filename}\n```\n{content}\n```")
                except Exception:
                    lines.append(f"### {filename}\n(could not read file)")
            file_contents = "\n\n".join(lines)

        prompt = (
            f"{WORKER_PROMPT.format(file_contents=file_contents)}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
