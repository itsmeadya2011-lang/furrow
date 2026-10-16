from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_READ_RE = re.compile(r"<<<READ:\s*([^\s>]+)>>>")
_WRITE_RE = re.compile(r"<<<WRITE:\s*([^\s>]+)>>>(.*?)<<<END>>>", re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Pre-read any files listed in the task for context
        context_parts: list[str] = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"--- {file_path} ---\n{content}\n--- end ---")
            except FileNotFoundError:
                context_parts.append(f"--- {file_path} ---\n(file does not exist yet)\n--- end ---")
            except Exception as e:
                context_parts.append(f"--- {file_path} ---\n(error reading: {e})\n--- end ---")

        context = "\n".join(context_parts)
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if context:
            prompt += f"\nExisting file contents:\n{context}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Execute any file operations requested by the LLM
        writes: list[tuple[str, str]] = []
        reads: list[str] = []

        for match in _READ_RE.finditer(response):
            reads.append(match.group(1))
        for match in _WRITE_RE.finditer(response):
            writes.append((match.group(1), match.group(2)))

        for path in reads:
            try:
                content = await self.client.read_file(path)
                response += f"\n\n[READ {path}]\n{content}\n[/READ {path}]"
            except Exception as e:
                response += f"\n\n[READ {path}] ERROR: {e}\n[/READ {path}]"

        for path, content in writes:
            try:
                await self.client.write_file(path, content)
                response += f"\n\n[WROTE {path}]"
            except Exception as e:
                response += f"\n\n[WRITE {path}] ERROR: {e}\n"

        return response.strip()
