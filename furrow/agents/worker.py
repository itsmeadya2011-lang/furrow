from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_CODE_BLOCK_PATTERN = re.compile(r'```(\w+)?\s*(.*?)\n(.*?)```', re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written: list[str] = []
        matches = _CODE_BLOCK_PATTERN.findall(response)
        for i, (_lang, filename_hint, code) in enumerate(matches):
            path = filename_hint.strip() if filename_hint.strip() else (
                self.task.files[0] if (i == 0 and self.task.files) else None
            )
            if not path:
                continue
            await self.client.write_file(path, code)
            written.append(path)

        if written:
            return f"Wrote {len(written)} file(s): {', '.join(written)}"
        return response
