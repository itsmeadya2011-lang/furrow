from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


FILE_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path (relative to workspace root).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path relative to workspace"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating parent directories as needed. Overwrites existing files.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files recursively under a directory relative to workspace root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path, default '.'"}},
            "required": [],
        },
    },
]


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def _handle_read_file(self, args: dict) -> str:
        return await self.client.read_file(args["path"])

    async def _handle_write_file(self, args: dict) -> str:
        await self.client.write_file(args["path"], args["content"])
        return "ok"

    async def _handle_list_files(self, args: dict) -> str:
        return "\n".join(self.client.list_files(args.get("path", ".")))

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        tool_handlers = {
            "read_file": self._handle_read_file,
            "write_file": self._handle_write_file,
            "list_files": self._handle_list_files,
        }
        return await self.client.complete_with_tools(
            prompt=prompt,
            tools=FILE_TOOLS,
            tool_handlers=tool_handlers,
            system="",
            model=self.client.settings.worker_model,
        )
