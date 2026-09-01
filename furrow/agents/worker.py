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
        workspace_files = self.client.list_files(self.client.settings.workspace)
        file_listing = "\n".join(workspace_files[:200]) if workspace_files else "(empty workspace)"
        if len(workspace_files) > 200:
            file_listing += f"\n... and {len(workspace_files) - 200} more files"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Target files: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Workspace files (top 200):\n{file_listing}\n\n"
            f"When you create or modify files, use this exact format:\n"
            f"WRITE: relative/path/to/file.py\n"
            f"<file content here>\n"
            f"---END---\n\n"
            f"Example:\n"
            f"WRITE: src/main.py\n"
            f"print('hello')\n"
            f"---END---\n\n"
            f"Return a concise summary of changes after the file blocks."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._apply_writes_and_summarize(response)

    async def _apply_writes_and_summarize(self, response: str) -> str:
        pattern = re.compile(r"^WRITE:\s*(.+?)\n(.*?)^---END---", re.MULTILINE | re.DOTALL)
        matches = list(pattern.finditer(response))
        if not matches:
            return response

        written_files: list[str] = []
        for match in matches:
            rel_path = match.group(1).strip()
            content = match.group(2).strip()
            try:
                await self.client.write_file(rel_path, content)
                written_files.append(rel_path)
            except Exception as e:
                written_files.append(f"{rel_path} (error: {e})")

        summary_lines = [f"Wrote {len(written_files)} file(s):"]
        summary_lines.extend(f"  - {f}" for f in written_files)
        summary_lines.append("")
        summary = "\n".join(summary_lines)
        return f"{summary}\n{response}"
