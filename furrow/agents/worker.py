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

    async def _read_context_files(self) -> str:
        if not self.task.files:
            return ""
        sections = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
            except Exception:
                continue
            sections.append(f"--- {path} ---\n{content}")
        if not sections:
            return ""
        return "\n\n".join(sections)

    async def _write_response_files(self, response: str) -> int:
        pattern = re.compile(r"```(\w+)?:([^\n]+)\n(.*?)```", re.DOTALL)
        count = 0
        for match in pattern.finditer(response):
            filepath = match.group(2).strip()
            content = match.group(3)
            if not filepath:
                continue
            await self.client.write_file(filepath, content)
            count += 1
        return count

    async def run(self) -> str:
        context = await self._read_context_files()
        prompt = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"
        response = await self.client.complete(
            prompt, system=WORKER_PROMPT, model=self.client.settings.worker_model
        )
        written = await self._write_response_files(response)
        if written:
            return f"Wrote {written} file(s) for task: {self.task.description}"
        return response.strip()
