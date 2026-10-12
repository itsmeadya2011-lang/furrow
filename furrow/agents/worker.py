from __future__ import annotations

import json
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
        sections: list[str] = [WORKER_PROMPT, "", f"Goal context: {self.task.description}", ""]
        if self.task.files:
            for path in self.task.files:
                try:
                    content = await self.client.read_file(path)
                except Exception:
                    continue
                sections.append(f"--- {path} ---\n{content}\n")
        else:
            sections.append("Files to touch: any")

        prompt = "\n".join(sections)
        data = await self.client.complete_json(prompt, model=self.client.settings.worker_model)
        return json.dumps(data)