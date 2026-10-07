from __future__ import annotations

import re
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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse structured file changes: FILE: <path>\nCONTENT:\n<content>
        pattern = r"FILE:\s*(.+?)\s*\nCONTENT:\s*\n(.*?)(?=\nFILE:|\nSUMMARY:|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)
        for rel_path, content in matches:
            path = self.client.settings.workspace / rel_path.strip()
            await self.client.write_file(path, content)

        # Extract summary if present
        summary_match = re.search(r"SUMMARY:\s*(.*)", response, re.DOTALL)
        return summary_match.group(1).strip() if summary_match else response.strip()
