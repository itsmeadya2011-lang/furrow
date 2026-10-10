from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read context files if specified
        context = ""
        if self.task.files:
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    context += f"\n--- {file_path} ---\n{content}\n"
                except FileNotFoundError:
                    context += f"\n--- {file_path} ---\n(File not found, will be created)\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nCurrent file contents:{context}\n"
        logger.debug("worker_started", task_id=self.task.id, description=self.task.description[:100])
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and apply file changes from the response
        changes = self._parse_file_changes(response)
        for file_path, content in changes.items():
            await self.client.write_file(file_path, content)
            logger.info("file_written", path=file_path, size=len(content))

        summary = self._summarize_changes(changes, response)
        logger.info("worker_completed", task_id=self.task.id, files_changed=len(changes))
        return summary

    def _parse_file_changes(self, response: str) -> dict[str, str]:
        """Parse file changes from worker response.
        
        Expected format:
        ```file:path/to/file.py
        file content here
        ```
        """
        changes = {}
        # Match code blocks with file:path format
        pattern = r"```file:([^\n]+)\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        for file_path, content in matches:
            changes[file_path.strip()] = content.strip()
        return changes

    def _summarize_changes(self, changes: dict[str, str], response: str) -> str:
        if changes:
            files = ", ".join(changes.keys())
            return f"Modified {len(changes)} file(s): {files}"
        return response[:500]  # Return truncated response if no file changes
