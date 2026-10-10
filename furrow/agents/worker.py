from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.files import read_file, write_file, list_files

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(self.client.settings.workspace)

    async def _read_existing_files(self) -> dict[str, str]:
        """Read contents of existing files listed in the task."""
        contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                contents[file_path] = await read_file(full_path)
        return contents

    def _parse_code_blocks(self, response: str) -> list[tuple[str, str]]:
        """Parse fenced code blocks with file paths as info-strings.
        Expected format: ```path/to/file.py\n<content>\n```
        Returns list of (file_path, content) tuples.
        """
        pattern = r"```(?:\w+\n)?(.+?)\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        result = []
        for file_path, content in matches:
            file_path = file_path.strip()
            content = content.rstrip()
            if file_path and content:
                result.append((file_path, content))
        return result

    async def run(self) -> str:
        existing_files = await self._read_existing_files()

        file_context = ""
        if existing_files:
            file_context = "\n\nExisting files:\n"
            for path, content in existing_files.items():
                file_context += f"\n--- {path} ---\n{content}\n"

        files_to_touch = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_to_touch}{file_context}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        code_blocks = self._parse_code_blocks(response)
        written_files = []

        for file_path, content in code_blocks:
            full_path = self.workspace / file_path
            await write_file(full_path, content)
            written_files.append(file_path)

        if written_files:
            return f"Wrote {len(written_files)} file(s): {', '.join(written_files)}"
        return response
