from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = self.client.settings
        self.workspace = Path(self.settings.workspace)
        self.max_rounds = 3

    async def run(self) -> str:
        """Execute the task using tool calls."""
        system_prompt = WORKER_PROMPT

        # Initial context
        context = await self._build_initial_context()

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        for round_num in range(self.max_rounds):
            response = await self.client.complete_with_tools(
                messages=messages,
                model=self.client.settings.worker_model,
                tools=self._get_tools(),
            )

            # Check if the response contains tool calls
            if isinstance(response, dict) and "tool_calls" in response:
                messages.append({"role": "assistant", "content": response.get("content", ""), "tool_calls": response["tool_calls"]})

                # Execute tool calls
                for tool_call in response["tool_calls"]:
                    result = await self._execute_tool(tool_call)
                    messages.append({"role": "tool", "content": result, "tool_call_id": tool_call["id"]})

                continue

            # No tool calls - task is done
            return str(response)

        return "Max rounds reached. Task may be incomplete."

    async def _build_initial_context(self) -> str:
        """Build initial context for the worker."""
        context = f"Task: {self.task.description}\n"
        context += f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        context += f"Workspace: {self.workspace}\n\n"

        # Read relevant files if specified
        if self.task.files:
            context += "Current file contents:\n"
            for file_path in self.task.files:
                full_path = self.workspace / file_path
                if full_path.exists():
                    try:
                        content = await self.client.read_file(full_path)
                        context += f"\n--- {file_path} ---\n{content}\n"
                    except (OSError, UnicodeDecodeError) as e:
                        context += f"\n--- {file_path} ---\nError reading: {e}\n"
                else:
                    context += f"\n--- {file_path} ---\nFile does not exist (will be created)\n"

        return context

    def _get_tools(self) -> list[dict[str, Any]]:
        """Return available tools for the worker."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path from workspace root"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path from workspace root"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Directory path", "default": "."},
                        },
                        "required": ["directory"],
                    },
                },
            },
        ]

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        function_name = tool_call["function"]["name"]
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            return f"Error: Invalid JSON in tool arguments: {tool_call['function']['arguments']}"

        try:
            if function_name == "read_file":
                path = self.workspace / args["path"]
                content = await self.client.read_file(path)
                return content
            elif function_name == "write_file":
                path = self.workspace / args["path"]
                await self.client.write_file(path, args["content"])
                return f"Successfully wrote to {args['path']}"
            elif function_name == "run_command":
                return await self._run_command(args["command"], args.get("timeout", 60))
            elif function_name == "list_files":
                directory = self.workspace / args["directory"]
                files = self.client.list_files(directory)
                return json.dumps(files)
            else:
                return f"Error: Unknown tool {function_name}"
        except (OSError, json.JSONDecodeError, ValueError) as e:
            return f"Error executing {function_name}: {e}"

    async def _run_command(self, command: str, timeout: int = 60) -> str:
        """Run a shell command and return output."""
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode() + stderr.decode()
                return f"Exit code: {proc.returncode}\n{output}"
            except asyncio.TimeoutError:
                proc.kill()
                return f"Command timed out after {timeout} seconds"
        except (OSError, ValueError) as e:
            return f"Error running command: {e}"