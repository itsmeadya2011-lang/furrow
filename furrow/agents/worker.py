from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        """Execute the task using LLM with tool access."""
        from furrow.agents.prompts import WORKER_PROMPT

        # Build context about the workspace
        workspace_files = self.client.list_files(self.client.settings.workspace)
        context = self._build_context(workspace_files)

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n\n"
            f"Workspace context:\n{context}\n\n"
            "You have access to the following tools:\n"
            "- read_file(path): Read a file's contents\n"
            "- write_file(path, content): Write content to a file\n"
            "- list_files(directory): List files in a directory\n\n"
            "To use a tool, respond with a JSON object:\n"
            '{"tool": "read_file", "args": {"path": "src/file.py"}}\n'
            '{"tool": "write_file", "args": {"path": "src/file.py", "content": "..."}}'
        )

        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                prompt, model=self.client.settings.worker_model
            )

            # Try to parse tool call
            try:
                tool_call = json.loads(response.strip())
                if isinstance(tool_call, dict) and "tool" in tool_call:
                    result = await self._execute_tool(tool_call)
                    prompt += f"\n\nTool result:\n{result}\n\nContinue or return final summary."
                    continue
            except json.JSONDecodeError:
                pass

            # If not a tool call, treat as final summary
            return response.strip()

        return "Max tool rounds reached. Task may be incomplete."

    def _build_context(self, files: list[str]) -> str:
        if not files:
            return "(empty workspace)"
        # Show first 50 files
        return "\n".join(files[:50])

    async def _execute_tool(self, tool_call: dict) -> str:
        tool = tool_call.get("tool")
        args = tool_call.get("args", {})

        if tool == "read_file":
            path = args.get("path")
            if not path:
                return "Error: path required"
            try:
                content = await self.client.read_file(path)
                return content
            except (FileNotFoundError, OSError) as e:
                return f"Error reading file: {e}"

        elif tool == "write_file":
            path = args.get("path")
            content = args.get("content")
            if not path or content is None:
                return "Error: path and content required"
            try:
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            except (OSError, IOError) as e:
                return f"Error writing file: {e}"

        elif tool == "list_files":
            directory = args.get("directory", ".")
            try:
                files = self.client.list_files(directory)
                return "\n".join(files)
            except (OSError, IOError) as e:
                return f"Error listing files: {e}"

        return f"Unknown tool: {tool}"