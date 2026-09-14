"""Worker agent that uses tool calls to edit files."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"You have access to these tools:\n"
            f"- read_file: Read a file's contents\n"
            f"- write_file: Write content to a file (creates parent dirs)\n"
            f"- list_files: List files in a directory\n\n"
            f"Use the tools to read existing files, then write the updated files. "
            f"When done, return a concise summary of what you changed."
        )

        tool_handlers = {
            "read_file": {
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"},
                    },
                    "required": ["path"],
                },
                "func": lambda path: self.client.read_file(path),
            },
            "write_file": {
                "description": "Write content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
                "func": lambda path, content: self.client.write_file(path, content),
            },
            "list_files": {
                "description": "List files in a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"},
                    },
                    "required": ["path"],
                },
                "func": lambda path: self.client.list_files(path),
            },
        }

        result = await self.client.complete_with_tools(
            prompt=prompt,
            system="You are a worker agent that edits files. Use tools to read and write files.",
            tool_handlers=tool_handlers,
            model=self.client.settings.worker_model,
        )
        self.task.result = result
        return result