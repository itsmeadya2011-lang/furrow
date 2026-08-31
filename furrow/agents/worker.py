from __future__ import annotations

import json
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
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_sections: list[str] = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_sections.append(f"--- {file_path} ---\n{content}")
            except (FileNotFoundError, OSError):
                file_sections.append(f"--- {file_path} --- (new file)")

        files_block = "\n\n".join(file_sections) if file_sections else "(no files specified)"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"{files_block}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            for change in data.get("changes", []):
                await self.client.write_file(change["file"], change["content"])
            return data.get("summary", "")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return response
