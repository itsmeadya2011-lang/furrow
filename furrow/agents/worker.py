from __future__ import annotations

import asyncio
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
        return await self.client.complete(prompt, model=self.client.settings.worker_model)

    async def read_file(self, path: str) -> str:
        return await self.client.read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        await self.client.write_file(path, content)

    def list_files(self, directory: str = ".") -> list[str]:
        return self.client.list_files(directory)

    async def execute_code(self, code: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return stdout.decode() + stderr.decode()
        except Exception as e:
            return str(e)
