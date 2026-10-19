from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_FILE_BLOCK_RE = re.compile(
    r"^FILE:\s*(.+?)\s*\n```(?:\w+)?\n(.*?)\n```\s*$",
    re.MULTILINE | re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read existing files so the worker can see current contents
        existing: dict[str, str] = {}
        for path in self.task.files:
            try:
                existing[path] = await self.client.read_file(path)
            except FileNotFoundError:
                existing[path] = ""
            except Exception:
                existing[path] = ""

        files_context = "\n".join(
            f"### {path}\n{existing[path]}" for path in self.task.files
        ) if self.task.files else "No existing files to read."

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Existing file contents:\n{files_context}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written = await self._parse_and_write(response)
        summary_lines = [f"Task {self.task.id}: wrote {len(written)} file(s)"]
        for path in written:
            summary_lines.append(f"  - {path}")
        return "\n".join(summary_lines)

    async def _parse_and_write(self, response: str) -> list[str]:
        """Parse FILE blocks from the LLM response and write them to disk.

        Returns a list of file paths that were written.
        """
        written: list[str] = []
        for match in _FILE_BLOCK_RE.finditer(response):
            path = match.group(1).strip()
            content = match.group(2)
            if not content.endswith("\n"):
                content += "\n"
            await self.client.write_file(path, content)
            written.append(path)
        return written
