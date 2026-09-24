from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = self.client.settings.workspace
        self.max_rounds = 3

    async def run(self) -> str:
        """Execute the task using tool calls."""
        conversation = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\nWorkspace: {self.workspace}"}
        ]
        
        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                self._format_conversation(conversation),
                model=self.client.settings.worker_model
            )
            
            # Try to parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return the response as result
                return response
            
            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "tool", "content": result, "name": tool_call.get("name", "unknown")})
        
        return "Max rounds reached. Task may be incomplete."

    def _format_conversation(self, conversation: list[dict]) -> str:
        """Format conversation for LLM completion."""
        parts = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool Result: {content}")
        return "\n\n".join(parts)

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from LLM response. Expects JSON format."""
        try:
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                if isinstance(data, dict) and "tool_calls" in data:
                    return data["tool_calls"]
                elif isinstance(data, list):
                    return data
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute a single tool call and return result."""
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        
        try:
            if name == "read_file":
                return await self._tool_read_file(args.get("path", ""))
            elif name == "write_file":
                return await self._tool_write_file(args.get("path", ""), args.get("content", ""))
            elif name == "list_files":
                return await self._tool_list_files(args.get("path", "."))
            elif name == "run_command":
                return await self._tool_run_command(args.get("command", []), args.get("cwd", ""))
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {e}"

    async def _tool_read_file(self, path: str) -> str:
        full_path = self.workspace / path
        if not full_path.exists():
            return f"File not found: {path}"
        content = await self.client.read_file(full_path)
        return f"File: {path}\n{content}"

    async def _tool_write_file(self, path: str, content: str) -> str:
        full_path = self.workspace / path
        await self.client.write_file(full_path, content)
        return f"Written to {path}"

    async def _tool_list_files(self, path: str) -> str:
        full_path = self.workspace / path
        files = self.client.list_files(full_path)
        return f"Files in {path}:\n" + "\n".join(files) if files else f"No files in {path}"

    async def _tool_run_command(self, command: list[str], cwd: str) -> str:
        work_dir = self.workspace / cwd if cwd else self.workspace
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode() + stderr.decode()
            return f"Exit code: {proc.returncode}\n{output}"
        except asyncio.TimeoutError:
            proc.kill()
            return "Command timed out after 60 seconds"
