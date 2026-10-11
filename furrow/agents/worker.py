from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging_config import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.workspace = Path(settings.workspace) if settings else Path.cwd()
        self.max_rounds = 5

    async def run(self) -> str:
        """Run the worker agent with tool-use loop."""
        # Read any files mentioned in the task to provide context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    file_contents[file_path] = content
                except Exception as e:
                    file_contents[file_path] = f"Error reading file: {e}"

        system_prompt = WORKER_PROMPT
        if file_contents:
            system_prompt += "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                system_prompt += f"\n--- {path} ---\n{content}\n"

        messages = [
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"}
        ]

        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                prompt=self._format_messages(messages),
                system=system_prompt,
                model=self.client.settings.worker_model
            )

            # Try to parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as final result
                return response

            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.get("id", "unknown")})

        return "Max rounds reached without completion"

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for the LLM."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<{role}>\n{content}\n</{role}>\n"
        return formatted

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        # Look for JSON tool calls in the response
        # Format: {"tool": "tool_name", "arguments": {...}, "id": "call_123"}
        import re
        pattern = r'\{"tool":\s*"([^"]+)",\s*"arguments":\s*(\{.*?\}),\s*"id":\s*"([^"]+)"\}'
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            tool_name, args_str, call_id = match
            try:
                args = json.loads(args_str)
                tool_calls.append({"name": tool_name, "arguments": args, "id": call_id})
            except json.JSONDecodeError:
                continue
        return tool_calls

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        try:
            if tool_name == "read_file":
                path = self.workspace / args.get("path", "")
                content = await self.client.read_file(path)
                return f"File content:\n{content}"
            elif tool_name == "write_file":
                path = self.workspace / args.get("path", "")
                content = args.get("content", "")
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            elif tool_name == "list_files":
                path = self.workspace / args.get("path", ".")
                files = self.client.list_files(path)
                return f"Files in {path}:\n" + "\n".join(files)
            elif tool_name == "run_command":
                cmd = args.get("command", [])
                cwd = self.workspace / args.get("cwd", ".")
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                return f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Tool execution error: {e}"
