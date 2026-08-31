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
        self.settings = settings

    async def run(self) -> str:
        files_context = ""
        if self.task.files:
            for f in self.task.files:
                try:
                    content = await self.client.read_file(f)
                    files_context += f"\n--- Current content of {f} ---\n{content}\n"
                except Exception:
                    files_context += f"\n--- {f} does not exist yet ---\n"

        workspace_files = ""
        if not self.task.files:
            try:
                all_files = self.client.list_files(self.settings.workspace if self.settings else ".")
                workspace_files = f"\nAvailable files in workspace:\n" + "\n".join(f"  - {f}" for f in all_files[:50])
            except Exception:
                pass

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"{files_context}{workspace_files}\n\n"
            "After implementing the task, output the changes in this format:\n"
            '<write path="relative/path/to/file">\n'
            "<![CDATA[\n"
            "file contents here\n"
            "]]>\n"
            "</write>\n\n"
            "Then provide a concise summary of what you changed and any issues."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        summary = await self._apply_writes(response)
        return summary

    async def _apply_writes(self, response: str) -> str:
        pattern = re.compile(
            r'<write\s+path="([^"]+)">\s*<!\[CDATA\[(.*?)\]\]>\s*</write>',
            re.DOTALL | re.IGNORECASE,
        )
        matches = pattern.findall(response)
        written = []
        for path, content in matches:
            try:
                await self.client.write_file(path, content.strip())
                written.append(path)
            except Exception as e:
                written.append(f"{path} (failed: {e})")
        summary = f"Wrote {len(matches)} file(s): {', '.join(written) if written else 'none'}"
        return summary
