from __future__ import annotations

import re
from pathlib import Path
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
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        changes = self._parse_changes(response)
        written: list[str] = []
        for file_path, content in changes.items():
            await self.client.write_file(Path(file_path), content)
            written.append(file_path)

        if written:
            summary = f"Wrote {len(written)} file(s): {', '.join(written)}."
        else:
            summary = "No file changes were written."

        return summary

    def _parse_changes(self, response: str) -> dict[str, str]:
        changes: dict[str, str] = {}
        pattern = re.compile(r"^###\s+(.+?)\s*\n```.*?\n(.*?)\n```", re.DOTALL | re.MULTILINE)
        for match in pattern.finditer(response):
            file_path = match.group(1).strip()
            content = match.group(2)
            changes[file_path] = content
        return changes
