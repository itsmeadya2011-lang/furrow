from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("worker")


WORKER_TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"}
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in a directory recursively",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list files from"}
            },
            "required": ["directory"],
        },
    },
]


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace_info = ""
        if self.task.files:
            workspace_info = f"\nRelevant files: {', '.join(self.task.files)}"
        else:
            workspace_info = "\nNo specific files specified. Explore the codebase as needed."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}{workspace_info}\n\nYou have access to file operations. Read files to understand the codebase, then write your changes. When done, provide a summary of what you changed."

        logger.info("worker_started", task_id=self.task.id)
        response = await self.client.complete_with_tools(
            prompt=prompt,
            model=self.client.settings.worker_model,
            tools=WORKER_TOOLS,
            max_rounds=15,
        )
        logger.info("worker_completed", task_id=self.task.id)
        return response
