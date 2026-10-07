from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

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

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        # Build context with file contents
        context_parts = [f"Task: {self.task.description}"]
        if file_contents:
            context_parts.append("\nExisting files:")
            for path, content in file_contents.items():
                context_parts.append(f"\n--- {path} ---\n{content}")
        else:
            context_parts.append("\nNo existing files to modify. Create new files as needed.")

        prompt = f"{WORKER_PROMPT}\n\n" + "\n".join(context_parts)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse response for file operations
        # Expected format: JSON with "files" array of {path, content} and "commands" array
        try:
            import json
            data = json.loads(response)
            if "files" in data:
                for file_op in data["files"]:
                    path = file_op.get("path")
                    content = file_op.get("content", "")
                    if path:
                        await self.client.write_file(self.workspace / path, content)
            if "commands" in data:
                for cmd in data["commands"]:
                    await self._run_command(cmd)
            return data.get("summary", "Task completed")
        except json.JSONDecodeError:
            # If not JSON, treat as plain text summary
            return response

    async def _run_command(self, cmd: str | list[str]) -> str:
        if isinstance(cmd, str):
            cmd = cmd.split()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode() + stderr.decode()
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{output}")
        return output
