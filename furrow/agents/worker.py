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

        written: list[str] = []
        errors: list[str] = []

        file_matches = list(re.finditer(r"^FILE:\s*(.+)$", response, re.MULTILINE))
        for i, match in enumerate(file_matches):
            path = match.group(1).strip()
            start = match.end()
            end = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(response)
            block = response[start:end]
            code_match = re.search(r"```(?:\w+\n)?(.*?)```", block, re.DOTALL)
            if code_match:
                content = code_match.group(1)
                try:
                    await self.client.write_file(path, content)
                    written.append(path)
                except Exception as exc:
                    errors.append(f"Failed to write {path}: {exc}")
            else:
                errors.append(f"No code block found for {path}")

        if written:
            summary = f"Wrote {len(written)} file(s): {', '.join(written)}"
        else:
            summary = "No files written"
        if errors:
            summary += f". Errors: {'; '.join(errors)}"
        return summary
