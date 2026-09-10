from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        log.info("task_start", task_id=self.task.id, description=self.task.description)
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"# File not found: {file_path}\n"

        # Build context from files
        context = ""
        if file_contents:
            context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse response for file operations
        await self._apply_changes(response)

        log.info("task_complete", task_id=self.task.id)
        return response

    async def _apply_changes(self, response: str) -> None:
        """Parse LLM response for file write operations and apply them."""
        # Look for code blocks with file paths
        # Format: ```python\n# file: path/to/file.py\ncode here\n```
        pattern = r"```(\w+)?\n# file: (.+?)\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        for lang, file_path, code in matches:
            file_path = file_path.strip()
            code = code.strip()
            if file_path and code:
                await self.client.write_file(file_path, code)
                log.info("file_written", file=file_path)

        # Also check for explicit write commands
        # Format: WRITE file: path/to/file.py\n```\ncontent\n```
        write_pattern = r"WRITE file: (.+?)\n```(\w+)?\n(.*?)\n```"
        write_matches = re.findall(write_pattern, response, re.DOTALL)
        for file_path, lang, code in write_matches:
            file_path = file_path.strip()
            code = code.strip()
            if file_path and code:
                await self.client.write_file(file_path, code)
                log.info("file_written", file=file_path)