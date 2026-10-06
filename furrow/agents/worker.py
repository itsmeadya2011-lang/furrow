from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("worker")


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
        self.settings = settings or self.client.settings
        self.workspace = workspace or self.settings.workspace
        self.max_tool_rounds = 3

    async def run(self) -> str:
        """Execute the task by using tools to read/write files."""
        logger.info("task_start", task_id=self.task.id, description=self.task.description)
        system_prompt = self._build_system_prompt()
        
        messages: list[dict[str, str]] = [
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"}
        ]

        for round_num in range(self.max_tool_rounds):
            logger.debug("llm_request", round=round_num + 1, task_id=self.task.id)
            response = await self.client.complete(
                self._format_messages(messages),
                system=system_prompt,
                model=self.client.settings.worker_model,
            )

            # Try to parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as summary
                logger.info("task_complete", task_id=self.task.id, summary=response[:200])
                return response

            logger.info("tool_calls_parsed", task_id=self.task.id, num_calls=len(tool_calls))
            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result: {result}"})

        logger.warning("max_tool_rounds_reached", task_id=self.task.id)
        return "Max tool rounds reached. Task may be incomplete."

    def _build_system_prompt(self) -> str:
        return f"""{WORKER_PROMPT}

You are working in the workspace: {self.workspace}

Available tools (respond with JSON only when using tools):
1. read_file: {{"tool": "read_file", "path": "relative/path/to/file"}}
2. write_file: {{"tool": "write_file", "path": "relative/path/to/file", "content": "file content"}}
3. list_files: {{"tool": "list_files", "directory": "relative/path"}}
4. run_command: {{"tool": "run_command", "command": ["cmd", "arg1", "arg2"], "cwd": "optional/working/dir"}}

Rules:
- Use tools to explore and modify the codebase
- Make minimal, targeted changes
- Return a concise summary when done (no tool calls)
- Do not use tools outside the workspace
- Prefer reading files before writing them
"""

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        """Format messages for the LLM."""
        formatted = []
        for msg in messages:
            formatted.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n\n".join(formatted)

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        # Try to find JSON objects in the response
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        tool_calls.append(parsed)
                except json.JSONDecodeError:
                    continue
        return tool_calls

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        tool = tool_call.get("tool")
        logger.debug("executing_tool", tool=tool, args={k: v for k, v in tool_call.items() if k != "tool"})
        try:
            if tool == "read_file":
                path = self.workspace / tool_call["path"]
                if not self._is_safe_path(path):
                    return f"Error: Path {tool_call['path']} is outside workspace"
                content = await self.client.read_file(path)
                logger.debug("file_read", path=tool_call["path"], size=len(content))
                return f"File content:\n{content}"

            elif tool == "write_file":
                path = self.workspace / tool_call["path"]
                if not self._is_safe_path(path):
                    return f"Error: Path {tool_call['path']} is outside workspace"
                content = tool_call.get("content", "")
                await self.client.write_file(path, content)
                logger.info("file_written", path=tool_call["path"], size=len(content))
                return f"Successfully wrote to {tool_call['path']}"

            elif tool == "list_files":
                directory = tool_call.get("directory", ".")
                path = self.workspace / directory
                if not self._is_safe_path(path):
                    return f"Error: Path {directory} is outside workspace"
                files = self.client.list_files(path)
                logger.debug("files_listed", directory=directory, count=len(files))
                return f"Files in {directory}:\n" + "\n".join(files)

            elif tool == "run_command":
                cmd = tool_call["command"]
                cwd = tool_call.get("cwd", ".")
                cwd_path = self.workspace / cwd
                if not self._is_safe_path(cwd_path):
                    return f"Error: Working directory {cwd} is outside workspace"
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
                    logger.debug("command_executed", command=cmd, exit_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("command_timeout", command=cmd)
                    return "Error: Command timed out after 60 seconds"

            else:
                logger.warning("unknown_tool", tool=tool)
                return f"Error: Unknown tool '{tool}'"

        except Exception as e:
            logger.error("tool_execution_error", tool=tool, error=str(e))
            return f"Error executing tool: {e}"

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is within workspace."""
        try:
            path.resolve().relative_to(self.workspace.resolve())
            return True
        except ValueError:
            return False