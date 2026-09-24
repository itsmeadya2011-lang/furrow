from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from rich.console import Console

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None, semaphore: asyncio.Semaphore | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.semaphore = semaphore

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for f in self.task.files:
            try:
                path = self.settings.workspace / f
                if path.exists():
                    file_contents[f] = await self.client.read_file(path)
            except Exception as e:  # noqa: BLE001
                file_contents[f] = f"Error reading: {e}"

        context = ""
        if file_contents:
            context = "\n\nCurrent file contents:\n" + "\n\n".join(
                f"--- {f} ---\n{content}" for f, content in file_contents.items()
            )

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to modify: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"

        if self.semaphore:
            async with self.semaphore:
                result = await self._execute_with_tools(prompt)
        else:
            result = await self._execute_with_tools(prompt)

        return result

    async def _execute_with_tools(self, prompt: str) -> str:
        """Execute the task by having the LLM produce a plan with tool calls, then execute them."""
        # For now, we'll use a simple approach: ask the LLM to produce the file changes directly
        # In a more advanced version, this could use function calling
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to parse file changes from the response
        # Expected format: ```file:path\ncontent\n``` for each file
        await self._apply_file_changes(response)

        return response

    async def _apply_file_changes(self, response: str) -> None:
        """Parse and apply file changes from LLM response."""

        # Pattern to match code blocks with file paths: ```file:path/to/file\ncontent\n```
        pattern = r"```file:([^\n]+)\n(.*?)\n```"
        matches = re.findall(pattern, response, re.DOTALL)

        for file_path, content in matches:
            file_path = file_path.strip()
            content = content.strip()
            try:
                full_path = self.settings.workspace / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                await self.client.write_file(full_path, content)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Error writing {file_path}: {e}[/red]")

    async def read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        return await self.client.read_file(self.settings.workspace / path)

    async def write_file(self, path: str, content: str) -> None:
        """Write a file to the workspace."""
        full_path = self.settings.workspace / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        await self.client.write_file(full_path, content)

    async def list_files(self, directory: str = ".") -> list[str]:
        """List files in the workspace."""
        return self.client.list_files(self.settings.workspace / directory)