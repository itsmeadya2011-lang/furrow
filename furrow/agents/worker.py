from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


logger = get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path.cwd()
        self.max_rounds = 10

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description[:100])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"},
        ]

        for round_num in range(self.max_rounds):
            logger.debug("worker_round", task_id=self.task.id, round=round_num)
            response = await self.client.complete(
                self._format_messages(messages),
                model=self.client.settings.worker_model,
            )

            # Try to parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as final answer
                logger.info("worker_completed", task_id=self.task.id, rounds=round_num + 1)
                return response

            # Execute tool calls
            for tool_call in tool_calls:
                logger.debug("tool_call", task_id=self.task.id, tool=tool_call["name"])
                result = await self._execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.get("id", "unknown")})

        logger.warning("worker_max_rounds_reached", task_id=self.task.id)
        return "Max rounds reached without completion"

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for completion API."""
        parts = []
        for msg in messages:
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

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response. Expected format: ```tool <name> <json>```"""
        tool_calls = []
        lines = response.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("```tool"):
                # Parse tool call
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    tool_name = parts[1]
                    try:
                        args = json.loads(parts[2])
                    except json.JSONDecodeError:
                        # Try to find the closing ```
                        json_str = parts[2]
                        j = i + 1
                        while j < len(lines) and not lines[j].strip().startswith("```"):
                            json_str += "\n" + lines[j]
                            j += 1
                        try:
                            args = json.loads(json_str)
                        except json.JSONDecodeError:
                            i = j
                            continue
                    tool_calls.append({"name": tool_name, "arguments": args, "id": f"call_{len(tool_calls)}"})
                i += 1
                # Skip to closing ```
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    i += 1
            i += 1
        return tool_calls

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        name = tool_call["name"]
        args = tool_call["arguments"]

        try:
            if name == "read_file":
                return await self._tool_read_file(args)
            elif name == "write_file":
                return await self._tool_write_file(args)
            elif name == "list_files":
                return await self._tool_list_files(args)
            elif name == "run_command":
                return await self._tool_run_command(args)
            elif name == "glob_files":
                return await self._tool_glob_files(args)
            else:
                return f"Error: Unknown tool '{name}'"
        except Exception as e:
            logger.error("tool_execution_failed", tool=name, error=str(e))
            return f"Error executing {name}: {e}"

    async def _tool_read_file(self, args: dict[str, Any]) -> str:
        path = args.get("path", "")
        if not path:
            return "Error: path is required"
        full_path = self.workspace / path
        if not full_path.exists():
            return f"Error: File not found: {path}"
        if not full_path.is_file():
            return f"Error: Not a file: {path}"
        content = await self.client.read_file(full_path)
        return f"File: {path}\n```\n{content}\n```"

    async def _tool_write_file(self, args: dict[str, Any]) -> str:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return "Error: path is required"
        full_path = self.workspace / path
        await self.client.write_file(full_path, content)
        logger.info("file_written", path=path)
        return f"Successfully wrote to {path}"

    async def _tool_list_files(self, args: dict[str, Any]) -> str:
        directory = args.get("directory", ".")
        full_path = self.workspace / directory
        files = self.client.list_files(full_path)
        if not files:
            return f"No files found in {directory}"
        return f"Files in {directory}:\n" + "\n".join(files)

    async def _tool_glob_files(self, args: dict[str, Any]) -> str:
        pattern = args.get("pattern", "**/*")
        full_path = self.workspace / pattern
        matches = list(full_path.parent.glob(full_path.name)) if "*" in full_path.name else []
        if hasattr(full_path, "glob"):
            matches = list(full_path.glob("**/*"))
        else:
            matches = list(self.workspace.glob(pattern))
        files = [str(f.relative_to(self.workspace)) for f in matches if f.is_file()]
        if not files:
            return f"No files matching pattern: {pattern}"
        return f"Files matching {pattern}:\n" + "\n".join(files)

    async def _tool_run_command(self, args: dict[str, Any]) -> str:
        command = args.get("command", "")
        if not command:
            return "Error: command is required"
        timeout = args.get("timeout", 60)

        # Security: only allow certain commands
        allowed_commands = ["pytest", "python", "npm", "pnpm", "yarn", "cargo", "go", "make", "uv", "pip", "node", "echo"]
        cmd_parts = shlex.split(command)
        if not cmd_parts or cmd_parts[0] not in allowed_commands:
            return f"Error: Command not allowed: {cmd_parts[0] if cmd_parts else 'empty'}"

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                output = stdout.decode() + stderr.decode()
                return f"Exit code: {proc.returncode}\n{output}"
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: Command timed out after {timeout}s"
        except FileNotFoundError:
            return f"Error: Command not found: {cmd_parts[0]}"
        except Exception as e:
            return f"Error: {e}"