from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

FILE_RE = re.compile(r"FILE:\s*(.+?)\s*\n\s*```(?:\w+)?\n([\s\S]*?)\n\s*```", re.MULTILINE)

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        context = ""
        if self.task.files:
            all_files = self.client.list_files(self.client.settings.workspace)
            relevant = [name for name in self.task.files if name in all_files][:3]
            for name in relevant:
                content = await self.client.read_file(self.client.settings.workspace / name)
                context += f"\n--- {name} ---\n{content}"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if context:
            prompt += f"\nExisting file context:\n{context}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written: list[str] = []
        for match in FILE_RE.finditer(response):
            path = match.group(1).strip()
            content = match.group(2)
            await self.client.write_file(path, content)
            written.append(path)

        if written:
            summary = self._extract_summary(response)
            return "Wrote files: " + ", ".join(written) + ("\n\n" + summary if summary else "")
        return response

    @staticmethod
    def _extract_summary(response: str) -> str:
        parts = response.rsplit("```", 1)
        return parts[-1].strip() if len(parts) > 1 else response.strip()
