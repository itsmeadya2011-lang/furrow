from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FILE_BLOCK_RE = re.compile(
    r"FILE:\s*(?P<path>[^\n]+?)\s*\n```[a-zA-Z0-9]*\n(?P<content>.*?)```",
    re.DOTALL,
)


def _parse_file_changes(response: str) -> dict[str, str]:
    changes: dict[str, str] = {}
    for match in _FILE_BLOCK_RE.finditer(response):
        path = match.group("path").strip()
        content = match.group("content")
        if content.endswith("\n"):
            content = content[:-1]
        changes[path] = content
    return changes


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def _read_context(self) -> str:
        files = self.task.files or []
        if not files:
            return ""
        parts: list[str] = []
        for f in files:
            try:
                content = await self.client.read_file(f)
            except FileNotFoundError:
                continue
            parts.append(f"--- {f} ---\n{content}\n")
        return "\n".join(parts)

    async def run(self) -> str:
        context = await self._read_context()
        file_list = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {file_list}\n\n"
        )
        if context:
            prompt += "Existing file contents:\n\n" + context + "\n\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        changes = _parse_file_changes(response)
        if not changes:
            return "No file changes were returned by the LLM."

        written: list[str] = []
        for path, content in changes.items():
            await self.client.write_file(path, content)
            written.append(path)

        summary = ", ".join(written)
        return f"Wrote changes to: {summary}"
