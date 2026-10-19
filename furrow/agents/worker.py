from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FILE_BLOCK_RE = re.compile(
    r"FILE:\s*(?P<path>\S+)\s*\n(?P<content>.*?)\nENDFILE",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        from furrow.config import settings as default_settings

        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> str:
        # Read existing files for context
        context_parts: list[str] = []
        for fpath in self.task.files:
            try:
                content = await self.client.read_file(fpath)
                context_parts.append(f"--- {fpath} (existing) ---\n{content}")
            except (FileNotFoundError, OSError):
                context_parts.append(f"--- {fpath} (new file) ---")

        context = "\n\n".join(context_parts) if context_parts else "(no existing files)"

        files_list = ", ".join(self.task.files) if self.task.files else "any relevant files"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n\n"
            f"Existing file contents:\n{context}\n\n"
            f"Respond with file changes using this format:\n"
            f"FILE: path/to/file.py\n"
            f"<full new file content>\n"
            f"ENDFILE\n\n"
            f"Output ONLY the FILE blocks, nothing else."
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and apply file changes
        written: list[str] = []
        for match in _FILE_BLOCK_RE.finditer(response):
            path = match.group("path").strip()
            content = match.group("content")
            await self.client.write_file(path, content)
            written.append(path)

        if not written:
            return f"No file changes detected in response. Raw output ({len(response)} chars): {response[:200]}..."

        summary = f"Modified {len(written)} file(s): {', '.join(written)}"
        return summary
