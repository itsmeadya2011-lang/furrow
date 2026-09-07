from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

WORKER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to read."}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. Creates parent directories if needed. Overwrites existing content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The path to the file to write."},
                "content": {"type": "string", "description": "The content to write to the file."}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in a directory tree recursively. Returns relative paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "The directory to list files in."}
            },
            "required": ["directory"]
        }
    },
    {
        "name": "run_command",
        "description": "Run a shell command and return stdout and stderr output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "cwd": {"type": "string", "description": "Working directory for the command. Defaults to workspace root."}
            },
            "required": ["command"]
        }
    },
    {
        "name": "grep_files",
        "description": "Search for a regex pattern in files within a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The regex pattern to search for."},
                "directory": {"type": "string", "description": "The directory to search in."}
            },
            "required": ["pattern", "directory"]
        }
    }
]


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or self.client.settings.workspace

    async def run(self) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Task: {self.task.description}\n"
                    f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
                    f"Workspace: {self.workspace}\n"
                ),
            }
        ]
        max_iter = self.client.settings.max_worker_iterations
        text = ""

        for iteration in range(1, max_iter + 1):
            text, tool_calls = await self.client.complete_with_tools(
                system=WORKER_PROMPT,
                messages=messages,
                tools=WORKER_TOOLS,
                model=self.client.settings.worker_model,
            )

            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                })
                for tc in tool_calls:
                    result = await self._execute_tool(tc["name"], tc["arguments"])
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc["id"],
                    })
            else:
                return text or "Task completed."

        return f"Max iterations ({max_iter}) reached. Last response: {text}"

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        logger.debug("worker.tool_call", name=name, arguments=arguments)
        try:
            if name == "read_file":
                return await self.client.read_file(arguments["path"])
            elif name == "write_file":
                await self.client.write_file(arguments["path"], arguments["content"])
                return f"File written: {arguments['path']}"
            elif name == "list_files":
                files = self.client.list_files(arguments["directory"])
                return "\n".join(files) if files else "No files found."
            elif name == "run_command":
                cwd = arguments.get("cwd", str(self.workspace))
                return await self.client.run_command(arguments["command"], cwd=cwd)
            elif name == "grep_files":
                return await self.client.grep_files(arguments["pattern"], arguments["directory"])
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            logger.error("worker.tool_error", name=name, error=str(e))
            return f"Error executing {name}: {e}"
