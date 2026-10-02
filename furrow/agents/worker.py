from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient, ToolSpec

if TYPE_CHECKING:
    from furrow.config import Settings

LIST_FILES_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "directory": {"type": "string", "description": "Directory path to list (relative to workspace)."},
    },
    "required": ["directory"],
}
READ_FILE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to read (relative to workspace)."},
    },
    "required": ["path"],
}
WRITE_FILE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write (relative to workspace)."},
        "content": {"type": "string", "description": "Full file content to write."},
    },
    "required": ["path", "content"],
}


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="list_files",
                description="List files recursively under a directory in the workspace.",
                parameters=LIST_FILES_SCHEMA,
            ),
            ToolSpec(
                name="read_file",
                description="Read the full contents of a file in the workspace.",
                parameters=READ_FILE_SCHEMA,
            ),
            ToolSpec(
                name="write_file",
                description="Write content to a file in the workspace, creating parent dirs as needed.",
                parameters=WRITE_FILE_SCHEMA,
            ),
        ]

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        system = (
            f"{WORKER_PROMPT}\n\n"
            f"Workspace root: {workspace}\n"
            "Always operate within this workspace. Resolve relative paths against it."
        )
        prompt = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )

        async def _handle(name: str, args: dict) -> str:
            if name == "list_files":
                directory = args.get("directory", ".")
                result = self.client.list_files(directory)
                if result:
                    return f"list_files({directory}) -> {len(result)} files:\n" + "\n".join(result)
                return f"list_files({directory}) -> (empty)"
            elif name == "read_file":
                path = args.get("path", "")
                content = await self.client.read_file(path)
                return f"read_file({path}) ->\n{content}"
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                await self.client.write_file(path, content)
                return f"write_file({path}) -> wrote {len(content)} chars"
            else:
                return f"Unknown tool: {name}"

        return await self.client.complete_with_tools(
            prompt,
            system,
            self.client.settings.worker_model,
            self._tools(),
            _handle,
        )
