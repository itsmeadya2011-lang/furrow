from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.workspace = settings.workspace if settings else Path.cwd()

    async def run(self) -> str:
        # First, read any relevant files to provide context
        context = await self._gather_context()

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nRelevant context:\n{context}\n\nReturn your actions and summary."

        logger.debug("worker_started", task_id=self.task.id, task_description=self.task.description[:100])
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and execute actions
        result = await self._execute_actions(response)
        logger.info("worker_completed", task_id=self.task.id)
        return result

    async def _gather_context(self) -> str:
        """Gather context by reading relevant files."""
        context_parts = []

        # Read files mentioned in the task
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"--- {file_path} ---\n{content}")
                except Exception as e:
                    context_parts.append(f"--- {file_path} ---\n[Error reading: {e}]")

        # Also list files in workspace for context
        try:
            files = self.client.list_files(self.workspace)
            context_parts.append(f"\nWorkspace files:\n" + "\n".join(files[:50]))
        except Exception:
            pass

        return "\n\n".join(context_parts) if context_parts else "No existing context."

    async def _execute_actions(self, response: str) -> str:
        """Parse and execute actions from the LLM response."""
        # Extract actions using regex
        action_pattern = r'ACTION:\s*(\w+)\((.*?)\)'
        actions = re.findall(action_pattern, response, re.DOTALL)

        results = []
        summary = ""

        for action_name, action_args in actions:
            try:
                if action_name == "read_file":
                    # Parse path from args
                    path_match = re.search(r'"([^"]+)"', action_args)
                    if path_match:
                        path = path_match.group(1)
                        content = await self.read_file(path)
                        results.append(f"Read {path}: {len(content)} chars")
                        logger.debug("file_read", path=path, size=len(content))
                    else:
                        results.append(f"Invalid read_file args: {action_args}")

                elif action_name == "write_file":
                    # Parse path and content from args
                    # Format: write_file("path", "content")
                    path_match = re.search(r'"([^"]+)"', action_args)
                    content_match = re.search(r'write_file\([^,]+,\s*(.+)\)', action_args, re.DOTALL)
                    if path_match and content_match:
                        path = path_match.group(1)
                        content = content_match.group(1).strip().strip('"\'')
                        # Handle escaped newlines
                        content = content.replace('\\n', '\n')
                        await self.write_file(path, content)
                        results.append(f"Wrote {path}")
                        logger.info("file_written", path=path, size=len(content))
                    else:
                        results.append(f"Invalid write_file args: {action_args}")

                elif action_name == "run_command":
                    # Parse command from args
                    # Format: run_command(["cmd", "arg"])
                    cmd_match = re.search(r'\[(.*?)\]', action_args, re.DOTALL)
                    if cmd_match:
                        cmd_str = cmd_match.group(1)
                        # Parse command array
                        cmd = [arg.strip().strip('"\'') for arg in cmd_str.split(',') if arg.strip()]
                        exit_code, stdout, stderr = await self.run_command(cmd)
                        results.append(f"Command {cmd}: exit={exit_code}, stdout={stdout[:200]}, stderr={stderr[:200]}")
                        logger.info("command_run", cmd=cmd, exit_code=exit_code)
                    else:
                        results.append(f"Invalid run_command args: {action_args}")

            except Exception as e:
                results.append(f"Error executing {action_name}: {e}")
                logger.error("action_failed", action=action_name, error=str(e))

        # Extract summary
        summary_match = re.search(r'SUMMARY:\s*(.+)', response, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
        else:
            summary = response.strip()

        action_summary = "\n".join(results) if results else "No actions executed"
        return f"{action_summary}\n\n{summary}"

    async def read_file(self, path: str | Path) -> str:
        """Read a file from the workspace."""
        full_path = self.workspace / path
        return await self.client.read_file(full_path)

    async def write_file(self, path: str | Path, content: str) -> None:
        """Write a file to the workspace."""
        full_path = self.workspace / path
        await self.client.write_file(full_path, content)

    async def run_command(self, cmd: list[str], cwd: str | Path | None = None, timeout: int = 120) -> tuple[int, str, str]:
        """Run a shell command and return (exit_code, stdout, stderr)."""
        work_dir = Path(cwd) if cwd else self.workspace
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", f"Command timed out after {timeout}s"
