from __future__ import annotations

import re

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings=None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        pattern = re.compile(r"FILE:\s*(?P<path>.+?)\n```[^\n]*\n(?P<content>.*?)```", re.DOTALL)
        matches = list(pattern.finditer(response))

        written: list[str] = []
        for match in matches:
            path = match.group("path").strip()
            content = match.group("content")
            await self.client.write_file(path, content)
            written.append(path)

        if matches:
            summary = response[:matches[0].start()].strip()
        else:
            summary = response.strip()

        if written:
            summary = f"Wrote files: {', '.join(written)}\n\n{summary}"

        return summary
