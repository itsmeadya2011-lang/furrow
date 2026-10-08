from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


TOOL_SCHEMA = {
    "read_file": {
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"}
            },
            "required": ["path"]
        }
    },
    "write_file": {
        "description": "Write content to a file (creates directories as needed)",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"}
            },
            "required": ["path", "content"]
        }
    },
    "list_files": {
        "description": "List all files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to list files from"}
            },
            "required": ["directory"]
        }
    }
}


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        workspace_files = self.client.list_files(self.client.settings.workspace)
        workspace_list = "\n".join(f"  {f}" for f in workspace_files[:50])

        prompt = WORKER_PROMPT.format(
            task_description=self.task.description,
            files=", ".join(self.task.files) if self.task.files else "any",
            dependencies=", ".join(self.task.dependencies) if self.task.dependencies else "none",
            workspace_files=workspace_list,
        )

        messages = [
            {"role": "system", "content": "You are a worker agent. Use tools to implement the task. Return tool calls as JSON."},
            {"role": "user", "content": prompt}
        ]

        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                self._format_messages(messages),
                model=self.client.settings.worker_model,
                system=""
            )

            # Check for tool calls in response
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                # No more tool calls, return the response as summary
                return response

            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.get("id", "")})

        return "Max rounds reached. Task may be incomplete."

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for completion API."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "tool":
                formatted.append(f"Tool result: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        return "\n\n".join(formatted)

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from LLM response."""
        # Look for JSON tool calls in the response
        tool_calls = []

        # Pattern 1: Explicit tool_call JSON
        pattern = r'```json\s*(\{.*?"tool_call".*?\})\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if "tool_call" in data:
                    tool_calls.append(data["tool_call"])
            except json.JSONDecodeError:
                pass

        # Pattern 2: Direct tool call format
        pattern2 = r'```json\s*(\{"name":\s*"(\w+)",\s*"arguments":\s*\{.*?\}\})\s*```'
        matches2 = re.findall(pattern2, response, re.DOTALL)
        for match in matches2:
            try:
                data = json.loads(match[0])
                if "name" in data and "arguments" in data:
                    tool_calls.append({"name": data["name"], "arguments": data["arguments"]})
            except json.JSONDecodeError:
                pass

        return tool_calls

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        try:
            if name == "read_file":
                path = args.get("path", "")
                content = await self.client.read_file(path)
                return f"File content:\n{content}"
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            elif name == "list_files":
                directory = args.get("directory", ".")
                files = self.client.list_files(directory)
                return f"Files in {directory}:\n" + "\n".join(f"  {f}" for f in files)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001 - report any tool error as a string
            return f"Error executing {name}: {e}"