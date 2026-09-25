from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.workspace = self.settings.workspace
        self.max_rounds = 3

    async def run(self) -> str:
        # Read relevant files for context
        context = await self._gather_context()

        for round_num in range(1, self.max_rounds + 1):
            prompt = self._build_prompt(context, round_num)
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)

            # Check if the response indicates completion
            if "TASK_COMPLETE" in response:
                return response.replace("TASK_COMPLETE", "").strip()

            # Execute any tool calls in the response
            context = await self._execute_tools(response, context)

        return f"Task completed after {self.max_rounds} rounds. Last response: {response[:500]}"

    def _build_prompt(self, context: str, round_num: int) -> str:
        return f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
Working directory: {self.workspace}

Context (files and previous actions):
{context}

Round {round_num}/{self.max_rounds}

You have access to these tools:
- READ_FILE(path): Read a file's contents
- WRITE_FILE(path, content): Write content to a file
- RUN_COMMAND(cmd): Run a shell command (use sparingly)
- LIST_FILES(dir): List files in a directory

To use a tool, respond with:
TOOL: READ_FILE(path)
TOOL: WRITE_FILE(path, content)
TOOL: RUN_COMMAND(cmd)
TOOL: LIST_FILES(dir)

When the task is complete, include "TASK_COMPLETE" in your final response with a summary.
"""

    async def _gather_context(self) -> str:
        parts = []
        if self.task.files:
            for f in self.task.files:
                path = self.workspace / f
                if path.exists():
                    try:
                        content = await self.client.read_file(path)
                        parts.append(f"--- {f} ---\n{content}")
                    except Exception:
                        pass
        return "\n\n".join(parts) if parts else "(no file context)"

    async def _execute_tools(self, response: str, context: str) -> str:
        lines = response.strip().split("\n")
        new_context = context

        for line in lines:
            line = line.strip()
            if line.startswith("TOOL: READ_FILE("):
                path = line[15:-1].strip()
                try:
                    content = await self.client.read_file(self.workspace / path)
                    new_context += f"\n\n--- {path} (read) ---\n{content}"
                except Exception as e:
                    new_context += f"\n\n--- {path} (read error) ---\n{e}"
            elif line.startswith("TOOL: WRITE_FILE("):
                # Parse: TOOL: WRITE_FILE(path, content)
                import re
                match = re.match(r'TOOL: WRITE_FILE\((.+?),\s*(.+)\)', line, re.DOTALL)
                if match:
                    path = match.group(1).strip()
                    content = match.group(2).strip()
                    try:
                        await self.client.write_file(self.workspace / path, content)
                        new_context += f"\n\n--- {path} (written) ---\n{content[:500]}"
                    except Exception as e:
                        new_context += f"\n\n--- {path} (write error) ---\n{e}"
            elif line.startswith("TOOL: RUN_COMMAND("):
                cmd = line[17:-1].strip()
                try:
                    proc = await asyncio.create_subprocess_shell(
                        cmd,
                        cwd=self.workspace,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    new_context += f"\n\n--- Command: {cmd} ---\n{output}"
                except Exception as e:
                    new_context += f"\n\n--- Command: {cmd} (error) ---\n{e}"
            elif line.startswith("TOOL: LIST_FILES("):
                dir_path = line[16:-1].strip() or "."
                try:
                    files = self.client.list_files(self.workspace / dir_path)
                    new_context += f"\n\n--- Files in {dir_path} ---\n" + "\n".join(files)
                except Exception as e:
                    new_context += f"\n\n--- Files in {dir_path} (error) ---\n{e}"

        return new_context
