from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM
from furrow.config import Provider, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Tool definitions for the worker agent.
FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the full contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path. Creates parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files in a directory recursively.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to list.",
                },
            },
            "required": ["directory"],
        },
    },
]

# OpenAI-format tool schema (function calling).
OPENAI_FILE_TOOLS: list[dict[str, Any]] = [
    {
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }
    for t in FILE_TOOLS
]


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        # Use tool calling so the worker can read/write files.
        return await self.client.complete_with_tools(
            prompt=prompt,
            system=WORKER_SYSTEM,
            tools=self._get_tool_schema(),
            model=self.client.settings.worker_model,
        )

    def _get_tool_schema(self) -> list[dict[str, Any]]:
        provider = self.client.settings.provider
        if provider == Provider.ANTHROPIC:
            return FILE_TOOLS
        # OpenAI and Ollama both use the OpenAI-style tool format with "parameters".
        return OPENAI_FILE_TOOLS
