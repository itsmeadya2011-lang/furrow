from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Pattern to extract file blocks from LLM responses:
# ```file: path/to/file.py
# <contents>
# ```
FILE_BLOCK_RE = re.compile(r"```file:\s*(\S+)\n(.*?)```", re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_str}\n"

        # Read existing files for context
        existing = [f for f in self.task.files if f]
        if existing:
            context = await self.client.read_files(existing)
            if context:
                prompt += f"\nExisting file contents:\n{context}\n"

        prompt += (
            "\nWrite file changes using this format:\n"
            "```file: path/to/file.py\n"
            "<exact file contents>\n"
            "```\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        written = await self._write_files(response)

        summary_parts = [f"Task {self.task.id}: {self.task.description}"]
        if written:
            summary_parts.append("Files written:")
            for path in written:
                summary_parts.append(f"  - {path}")
        else:
            summary_parts.append("No files were written.")
        return "\n".join(summary_parts)

    async def _write_files(self, response: str) -> list[str]:
        """Parse file blocks from the LLM response and write them to disk."""
        written: list[str] = []
        for match in FILE_BLOCK_RE.finditer(response):
            path = match.group(1).strip()
            content = match.group(2)
            try:
                await self.client.write_file(path, content)
                written.append(path)
            except Exception:
                pass
        return written
