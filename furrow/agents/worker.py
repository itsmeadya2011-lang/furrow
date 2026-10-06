from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or global_settings)
        self.settings = self.client.settings

    async def run(self) -> str:
        # Read all files mentioned in the task
        file_contents = {}
        for file_path in self.task.files:
            path = Path(file_path)
            if path.exists():
                file_contents[file_path] = await self.client.read_file(path)
            else:
                file_contents[file_path] = ""  # New file

        # Build prompt with file contents
        files_section = ""
        if file_contents:
            files_section = "\n\nCurrent file contents:\n"
            for fname, content in file_contents.items():
                files_section += f"\n--- {fname} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_section}\n\nReturn your changes as JSON with this shape:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"Brief summary of changes\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse response and apply changes
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)

            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract code blocks
            return await self._fallback_apply(response, file_contents)

    async def _fallback_apply(self, response: str, file_contents: dict[str, str]) -> str:
        """Try to extract code blocks from response and apply them."""
        # Look for ```python or ```code blocks
        pattern = r"```(?:python|py)?\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        if matches and self.task.files:
            # Apply first match to first file
            await self.client.write_file(self.task.files[0], matches[0].strip())
            return f"Applied code block to {self.task.files[0]}"

        return f"Task completed (fallback): {response[:200]}"
