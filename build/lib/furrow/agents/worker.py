from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.workspace = Path(self.settings.workspace)
        self.max_rounds = 3

    async def run(self) -> str:
        """Execute the task using LLM with tool access for file operations."""
        # Read relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    file_contents[file_path] = content
                except Exception as e:
                    file_contents[file_path] = f"Error reading file: {e}"

        # Build initial context
        context = self._build_context(file_contents)

        for round_num in range(self.max_rounds):
            prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nContext:\n{context}\n\nRound {round_num + 1}/{self.max_rounds}.\nReturn JSON with your actions."
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)

            try:
                action = json.loads(response)
            except json.JSONDecodeError:
                return f"Worker completed: {response}"

            # Execute the action
            result = await self._execute_action(action)
            if result.get("done"):
                return result.get("summary", "Task completed")

            # Update context with action result
            context = self._build_context(file_contents, action_result=result)

        return f"Worker completed after {self.max_rounds} rounds"

    def _build_context(self, file_contents: dict[str, str], action_result: dict | None = None) -> str:
        parts = []
        if file_contents:
            parts.append("Current files:")
            for path, content in file_contents.items():
                parts.append(f"\n--- {path} ---\n{content}")
        if action_result:
            parts.append(f"\nLast action result: {action_result.get('summary', '')}")
        return "\n".join(parts)

    async def _execute_action(self, action: dict) -> dict:
        action_type = action.get("action")

        if action_type == "read_file":
            path = action.get("path")
            full_path = self.workspace / path
            if full_path.exists():
                content = await self.client.read_file(full_path)
                return {"done": False, "summary": f"Read {path}", "content": content}
            return {"done": False, "summary": f"File not found: {path}", "content": ""}

        elif action_type == "write_file":
            path = action.get("path")
            content = action.get("content", "")
            full_path = self.workspace / path
            await self.client.write_file(full_path, content)
            return {"done": False, "summary": f"Wrote {path}"}

        elif action_type == "list_files":
            directory = action.get("directory", ".")
            full_path = self.workspace / directory
            files = self.client.list_files(full_path)
            return {"done": False, "summary": f"Listed {directory}", "files": files}

        elif action_type == "run_command":
            cmd = action.get("command")
            if not cmd:
                return {"done": False, "summary": "No command provided"}
            import asyncio
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                return {"done": False, "summary": f"Ran: {cmd}", "output": output}
            except Exception as e:
                return {"done": False, "summary": f"Command failed: {e}", "output": str(e)}

        elif action_type == "complete":
            return {"done": True, "summary": action.get("summary", "Task completed")}

        else:
            return {"done": False, "summary": f"Unknown action: {action_type}"}
