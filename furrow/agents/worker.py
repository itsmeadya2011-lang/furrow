from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


# Matches "FILE: path" followed by a fenced code block until the next FILE: or end of string.
_FILE_BLOCK = re.compile(
    r"FILE:\s*(?P<path>\S+?)\s*\n(?P<content>.*?)(?=FILE:|\Z)",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_hint = ", ".join(self.task.files) if self.task.files else "any relevant files"

        # Read existing files so the LLM can see current contents before rewriting.
        existing: dict[str, str] = {}
        for path in self.task.files:
            try:
                existing[path] = await self.client.read_file(path)
            except FileNotFoundError:
                pass

        context = ""
        if existing:
            context = "\n\nExisting file contents:\n" + "\n\n".join(
                f"--- {path} ---\n{content}" for path, content in existing.items()
            )

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_hint}\n"
            f"{context}\n\n"
            f"Write the complete file contents using this exact format for each file:\n"
            f"FILE: path/to/file\n"
            f"```\n"
            f"<file contents>\n"
            f"```\n\n"
            f"Include every file you create or modify, with its FULL contents."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        written = await self._write_files(response)
        if written:
            return f"Wrote {len(written)} file(s): {', '.join(written)}"
        return f"Task completed. No files written. Response: {response[:200]}"

    async def _write_files(self, response: str) -> list[str]:
        written: list[str] = []
        for match in _FILE_BLOCK.finditer(response):
            path = match.group("path").strip()
            content = match.group("content").strip()
            # Strip surrounding code fences if present.
            fence = re.match(r"^```[\w-]*\n(.*)```\s*$", content, re.DOTALL)
            if fence:
                content = fence.group(1)
            await self.client.write_file(path, content)
            written.append(path)
        return written