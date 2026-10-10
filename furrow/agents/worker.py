from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


FILE_BLOCK_PATTERN = re.compile(r'```file:([^\n]+)\n(.*?)```', re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_contents = {}
        for file_path in self.task.files or []:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""

        context_parts = []
        for file_path, content in file_contents.items():
            context_parts.append(f"--- File: {file_path} ---\n{content}")

        context = "\n\n".join(context_parts) if context_parts else "(no existing files provided)"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting file contents:\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        matches = FILE_BLOCK_PATTERN.findall(response)
        changed_files = []
        for filename, content in matches:
            filename = filename.strip()
            content = content.rstrip()
            await self.client.write_file(filename, content)
            changed_files.append(filename)

        if changed_files:
            return f"Modified files: {', '.join(changed_files)}"
        return response.strip()
