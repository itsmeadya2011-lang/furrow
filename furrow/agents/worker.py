from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Build the prompt with task details
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
        if self.task.files:
            prompt += f"Files to touch: {', '.join(self.task.files)}\n"
            # Include file contents for context
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    prompt += f"\n--- {file_path} ---\n{content}\n"
                except FileNotFoundError:
                    prompt += f"\n--- {file_path} ---\n(File does not exist)\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # The worker should return a JSON with actions to perform
        try:
            actions = json.loads(response)
            if isinstance(actions, dict) and "actions" in actions:
                return await self._execute_actions(actions["actions"])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return the raw response as summary
        return response

    async def _execute_actions(self, actions: list[dict]) -> str:
        """Execute a list of file operations and commands."""
        results = []
        for action in actions:
            action_type = action.get("type")
            try:
                if action_type == "write_file":
                    path = action.get("path")
                    content = action.get("content", "")
                    await self.client.write_file(path, content)
                    results.append(f"Wrote {path}")
                elif action_type == "read_file":
                    path = action.get("path")
                    content = await self.client.read_file(path)
                    results.append(f"Read {path} ({len(content)} chars)")
                elif action_type == "run_command":
                    cmd = action.get("command", [])
                    result = await self._run_command(cmd)
                    results.append(f"Ran {' '.join(cmd)}: {result}")
                else:
                    results.append(f"Unknown action type: {action_type}")
            except Exception as e:
                results.append(f"Error executing {action_type}: {e}")
        return "\n".join(results)

    async def _run_command(self, cmd: list[str]) -> str:
        """Run a shell command and return output."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.settings.workspace,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            return (stdout.decode() + stderr.decode()).strip()
        except asyncio.TimeoutError:
            proc.kill()
            return "Command timed out"
