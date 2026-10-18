from __future__ import annotations
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

import structlog

logger = structlog.get_logger("furrow.worker")

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker.start", task_id=self.task.id)
        # Read files that need to be modified
        file_contents = {}
        files_to_read = self.task.files if self.task.files else []

        if not files_to_read:
            # No files specified - list the workspace to understand context
            files_to_read = self.client.list_files(self.client.settings.workspace)[:10]

        for f in files_to_read:
            try:
                file_contents[f] = await self.client.read_file(f)
            except (FileNotFoundError, IsADirectoryError):
                file_contents[f] = "(file does not exist yet)"

        # Build prompt with file contents
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
        if file_contents:
            prompt += "\nCurrent file contents:\n"
            for path, content in file_contents.items():
                prompt += f"\n--- {path} ---\n{content}\n"

        prompt += "\nReturn the complete updated file contents for each file that needs changes. Use this format:\n"
        prompt += "```filename: path/to/file\n<complete file content>\n```\n"
        prompt += "Only include files that actually need changes.\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and apply changes
        changed_files = await self._apply_changes(response)

        if changed_files:
            logger.info("worker.complete", task_id=self.task.id)
            return f"Completed task: {self.task.description}. Modified files: {', '.join(changed_files)}"
        logger.info("worker.complete", task_id=self.task.id)
        return f"Completed task: {self.task.description}. No file changes detected in response."

    async def _apply_changes(self, response: str) -> list[str]:
        """Parse the LLM response for file changes and write them."""
        pattern = r'```filename:\s*(.+?)\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        changed = []
        for filepath, content in matches:
            filepath = filepath.strip()
            await self.client.write_file(filepath, content)
            changed.append(filepath)
        return changed
