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
        blocks = re.findall(r'```(\w+)?\n(.*?)```', response, re.DOTALL)
        written_files: list[str] = []
        for index, (lang, code) in enumerate(blocks):
            path = None
            lines = code.split('\n', 1)
            first_line = lines[0].strip()
            if first_line.startswith('# filename:') or first_line.startswith('// filename:'):
                path = first_line.split(':', 1)[1].strip()
                code = lines[1] if len(lines) > 1 else ''
            if not path and self.task.files and index < len(self.task.files):
                path = self.task.files[index]
            if not path:
                continue
            await self.client.write_file(path, code)
            written_files.append(path)
        return f"Wrote {len(written_files)} files: {', '.join(written_files)}"
