from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self) -> str:
        # Get workspace context
        workspace_files = self.client.list_files(self.client.settings.workspace)

        # Build prompt with workspace context
        prompt = f"{WORKER_PROMPT}\n\n"
        prompt += f"Task: {self.task.description}\n"
        prompt += f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        prompt += f"Workspace files: {', '.join(workspace_files[:50])}\n"

        # Ask LLM to produce a plan of file operations
        prompt += """

Return JSON only with this shape:
{
  "operations": [
    {"type": "read", "path": "src/file.py"},
    {"type": "write", "path": "src/file.py", "content": "new content"},
    {"type": "exec", "command": "pytest -q"}
  ],
  "summary": "Brief summary of changes"
}
"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            operations = data.get("operations", [])
            summary = data.get("summary", "Task completed")

            # Execute operations
            for op in operations:
                await self._execute_operation(op)

            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: just return the raw response
            return f"Worker completed (raw): {response}"

    async def _execute_operation(self, op: dict) -> None:
        op_type = op.get("type")
        path = op.get("path")

        if op_type == "read":
            if path:
                content = await self.client.read_file(path)
                console_print(f"Read {path} ({len(content)} chars)")
        elif op_type == "write":
            if path:
                content = op.get("content", "")
                await self.client.write_file(path, content)
                console_print(f"Wrote {path} ({len(content)} chars)")
        elif op_type == "exec":
            command = op.get("command", "")
            if command:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.client.settings.workspace
                )
                stdout, stderr = await proc.communicate()
                output = stdout.decode() + stderr.decode()
                console_print(f"Executed: {command}\n{output}")
        else:
            console_print(f"Unknown operation type: {op_type}")


def console_print(msg: str) -> None:
    from rich.console import Console
    Console().print(f"[dim]{msg}[/dim]")
