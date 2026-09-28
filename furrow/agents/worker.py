from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient, Tool

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Executes a single task using LLM-powered tool calls."""

    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        return await self.client.run_tool_loop(
            prompt=prompt,
            tools=self._get_tools(),
            model=self.client.settings.worker_model,
        )

    def _get_tools(self) -> list[Tool]:
        """Return the file-system and command tools available to the worker."""
        return [
            Tool(
                name="read_file",
                description="Read the contents of a file at the given path.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read."},
                    },
                    "required": ["path"],
                },
                func=self._read_file,
            ),
            Tool(
                name="write_file",
                description=(
                    "Write content to a file. Creates parent directories if needed. "
                    "Overwrites existing content."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to write."},
                        "content": {"type": "string", "description": "Content to write to the file."},
                    },
                    "required": ["path", "content"],
                },
                func=self._write_file,
            ),
            Tool(
                name="list_files",
                description="List all files in a directory recursively.",
                parameters={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "Directory to list files in."},
                    },
                    "required": ["directory"],
                },
                func=self._list_files,
            ),
            Tool(
                name="run_command",
                description=(
                    "Execute a shell command in the workspace and return its output. "
                    "Use this for running tests, linting, or any project command."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute."},
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds before killing the command.",
                            "default": 60,
                        },
                    },
                    "required": ["command"],
                },
                func=self._run_command,
            ),
        ]

    # -- Tool implementations ------------------------------------------------

    async def _read_file(self, path: str) -> str:
        return await self.client.read_file(path)

    async def _write_file(self, path: str, content: str) -> str:
        await self.client.write_file(path, content)
        return f"File written successfully: {path}"

    async def _list_files(self, directory: str) -> str:
        files = self.client.list_files(directory)
        return "\n".join(files) if files else "No files found."

    async def _run_command(self, command: str, timeout: int = 60) -> str:
        return await self.client.run_command(command, timeout)
