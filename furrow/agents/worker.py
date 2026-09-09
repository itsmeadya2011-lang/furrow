from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Agent responsible for executing individual implementation tasks."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the worker with a task to execute.

        Args:
            task: The task definition containing description and target files.
            client: Optional LLM client instance.
            settings: Optional application settings.
        """
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        """
        Execute the assigned task.

        Sends the task to the LLM, parses any file content blocks from the
        response, writes them to disk, and returns a summary.

        Returns:
            A summary string of the work performed.
        """
        files_list = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to create/modify: {files_list}\n\n"
            "Provide the complete contents for each file you create or modify. "
            "For each file, use the following format:\n"
            "FILE: path/to/file.py\n"
            "<complete file content>\n"
            "Start each file block with 'FILE:' on its own line."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written: list[str] = []
        for path_str, content in self._parse_file_blocks(response):
            await self.client.write_file(path_str, content)
            written.append(path_str)

        summary_parts = [f"Worker completed task '{self.task.id}'."]
        if written:
            summary_parts.append(f"Wrote files: {', '.join(written)}")
        summary_parts.append(f"LLM notes: {response[:500]}")

        return "\n".join(summary_parts)

    def _parse_file_blocks(self, text: str) -> list[tuple[str, str]]:
        """
        Parse FILE: blocks from LLM response text.

        Args:
            text: Raw LLM response text.

        Returns:
            A list of (file_path, file_content) tuples.
        """
        pattern = re.compile(r"^FILE:\s*(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        blocks: list[tuple[str, str]] = []
        for i, match in enumerate(matches):
            path = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip("\n")
            blocks.append((path, content))
        return blocks
