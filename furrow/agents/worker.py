from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings as SettingsType


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: SettingsType | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self) -> str:
        # Read any relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"# File not found: {file_path}\n# Will create new file"

        files_context = "\n\n".join(
            [f"=== {path} ===\n{content}" for path, content in file_contents.items()]
        )

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles to work on:\n{files_context if files_context else 'No existing files - create new files as needed'}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse response for file operations
        await self._apply_changes(response)

        return response

    async def _apply_changes(self, response: str) -> None:
        """Parse the LLM response for file write operations and apply them."""
        # Pattern to match ```python\n# filepath: path/to/file.py\ncode```
        pattern = r"```(?:\w+)?\n# filepath: (.+?)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        for file_path, content in matches:
            file_path = file_path.strip()
            content = content.strip()
            if file_path and content:
                await self.client.write_file(file_path, content)
