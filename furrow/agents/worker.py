from __future__ import annotations

import asyncio
import json
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

    async def run(self) -> str:
        # Build context with relevant files
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}{files_context}\n\nYou have access to file operations. Return JSON with your actions:"
        prompt += """
{
  "actions": [
    {"type": "read", "path": "path/to/file"},
    {"type": "write", "path": "path/to/file", "content": "file content"},
    {"type": "run", "command": "pytest -q"}
  ],
  "summary": "What you did"
}
"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._execute_actions(response)

    async def _execute_actions(self, response: str) -> str:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # If not JSON, assume it's a summary
            return response

        actions = data.get("actions", [])
        summary = data.get("summary", "Task completed")

        results = []
        for action in actions:
            action_type = action.get("type")
            if action_type == "read":
                path = action.get("path")
                try:
                    content = await self.client.read_file(path)
                    results.append(f"Read {path}: {len(content)} chars")
                except Exception as e:
                    results.append(f"Failed to read {path}: {e}")
            elif action_type == "write":
                path = action.get("path")
                content = action.get("content", "")
                try:
                    await self.client.write_file(path, content)
                    results.append(f"Wrote {path}")
                except Exception as e:
                    results.append(f"Failed to write {path}: {e}")
            elif action_type == "run":
                command = action.get("command")
                try:
                    proc = await asyncio.create_subprocess_shell(
                        command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.client.settings.workspace
                    )
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    results.append(f"Ran '{command}':\n{output[:500]}")
                except asyncio.TimeoutError:
                    results.append(f"Command '{command}' timed out")
                except Exception as e:
                    results.append(f"Failed to run '{command}': {e}")

        return f"{summary}\n\nActions:\n" + "\n".join(results)
