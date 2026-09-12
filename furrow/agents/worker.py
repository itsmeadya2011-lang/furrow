from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path.cwd()
        self.max_rounds = 10
        self.round = 0
        self.logger = get_logger("worker")

    async def run(self) -> str:
        """Run the worker agent with tool use loop."""
        self.logger.info("worker_start", task_id=self.task.id, description=self.task.description)
        
        # Build initial context
        file_context = await self._get_file_context()
        
        messages = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"{WORKER_TOOLS_PROMPT}\n\nTask: {self.task.description}\nFiles to consider: {', '.join(self.task.files) if self.task.files else 'any'}\n\nWorkspace files:\n{file_context}"}
        ]

        while self.round < self.max_rounds:
            self.round += 1
            self.logger.debug("worker_round", task_id=self.task.id, round=self.round)
            
            response = await self.client.complete(
                self._format_messages(messages),
                model=self.client.settings.worker_model
            )
            
            # Try to parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, treat as final response
                self.logger.info("worker_complete", task_id=self.task.id, rounds=self.round)
                return response
            
            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                tool_results.append({"tool": tool_call["name"], "result": result})
            
            # Add tool results to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": json.dumps(tool_results, indent=2)})
        
        self.logger.warning("worker_max_rounds", task_id=self.task.id, rounds=self.round)
        return "Max rounds reached. Task may be incomplete."

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for LLM completion."""
        return "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from LLM response. Expects JSON format."""
        try:
            # Look for JSON tool calls in the response
            # Format: {"tool": "name", "args": {...}}
            lines = response.strip().split('\n')
            tool_calls = []
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        if "tool" in data and "args" in data:
                            tool_calls.append(data)
                    except json.JSONDecodeError:
                        continue
            return tool_calls
        except Exception:
            return []

    async def _execute_tool(self, tool_call: dict) -> Any:
        """Execute a tool call and return the result."""
        tool_name = tool_call["tool"]
        args = tool_call["args"]
        
        self.logger.debug("tool_call", tool=tool_name, args=args)
        
        try:
            if tool_name == "read_file":
                return await self._read_file(args.get("path", ""))
            elif tool_name == "write_file":
                return await self._write_file(args.get("path", ""), args.get("content", ""))
            elif tool_name == "list_files":
                return await self._list_files(args.get("path", "."))
            elif tool_name == "run_command":
                return await self._run_command(args.get("command", ""), args.get("cwd", "."))
            elif tool_name == "delete_file":
                return await self._delete_file(args.get("path", ""))
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error("tool_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    async def _read_file(self, path: str) -> str:
        """Read a file from the workspace."""
        full_path = self.workspace / path
        if not full_path.exists():
            return f"File not found: {path}"
        if not full_path.is_file():
            return f"Not a file: {path}"
        try:
            content = full_path.read_text(encoding="utf-8")
            self.logger.debug("file_read", path=path, size=len(content))
            return f"File: {path}\n{content}"
        except Exception as e:
            self.logger.error("file_read_error", path=path, error=str(e))
            return f"Error reading {path}: {e}"

    async def _write_file(self, path: str, content: str) -> str:
        """Write a file to the workspace."""
        full_path = self.workspace / path
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            self.logger.info("file_written", path=path, size=len(content))
            return f"Successfully wrote {path} ({len(content)} bytes)"
        except Exception as e:
            self.logger.error("file_write_error", path=path, error=str(e))
            return f"Error writing {path}: {e}"

    async def _list_files(self, path: str) -> str:
        """List files in a directory."""
        full_path = self.workspace / path
        if not full_path.exists():
            return f"Directory not found: {path}"
        if not full_path.is_dir():
            return f"Not a directory: {path}"
        try:
            files = []
            for f in full_path.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(self.workspace)
                    files.append(str(rel))
            result = "\n".join(files) if files else "No files found"
            self.logger.debug("files_listed", path=path, count=len(files))
            return result
        except Exception as e:
            self.logger.error("file_list_error", path=path, error=str(e))
            return f"Error listing {path}: {e}"

    async def _run_command(self, command: str, cwd: str) -> str:
        """Run a shell command."""
        full_cwd = self.workspace / cwd
        self.logger.info("command_run", command=command, cwd=cwd)
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=full_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode() + stderr.decode()
            self.logger.debug("command_complete", command=command, exit_code=proc.returncode, output_size=len(output))
            return f"Exit code: {proc.returncode}\n{output}"
        except asyncio.TimeoutError:
            self.logger.warning("command_timeout", command=command)
            return "Command timed out (60s)"
        except Exception as e:
            self.logger.error("command_error", command=command, error=str(e))
            return f"Error running command: {e}"

    async def _delete_file(self, path: str) -> str:
        """Delete a file."""
        full_path = self.workspace / path
        try:
            if full_path.exists():
                full_path.unlink()
                self.logger.info("file_deleted", path=path)
                return f"Deleted {path}"
            return f"File not found: {path}"
        except Exception as e:
            self.logger.error("file_delete_error", path=path, error=str(e))
            return f"Error deleting {path}: {e}"

    async def _get_file_context(self) -> str:
        """Get a summary of the workspace files for context."""
        try:
            files = []
            for f in self.workspace.rglob("*"):
                if f.is_file() and not any(part.startswith('.') for part in f.parts):
                    rel = f.relative_to(self.workspace)
                    size = f.stat().st_size
                    files.append(f"{rel} ({size} bytes)")
            return "\n".join(files[:50]) if files else "No files in workspace"
        except Exception:
            return "Could not read workspace"
