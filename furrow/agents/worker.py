from __future__ import annotations

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
        self.task.status = "in_progress"

    async def run(self) -> str:
        # First, read any relevant files to understand context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""

        # Build context from files
        context = ""
        if file_contents:
            context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}\n\nReturn a JSON object with:\n- \"changes\": list of {{ \"path\": str, \"content\": str }} for files to create/modify\n- \"commands\": list of shell commands to run (optional)\n- \"summary\": brief summary of what was done"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: treat as plain text summary
            return response

        # Apply file changes
        changes = data.get("changes", [])
        for change in changes:
            path = change.get("path")
            content = change.get("content", "")
            if path:
                await self.client.write_file(path, content)

        # Run commands
        commands = data.get("commands", [])
        results = []
        for cmd in commands:
            try:
                import asyncio
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.client.settings.workspace,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                results.append(f"$ {cmd}\n{stdout.decode()}\n{stderr.decode()}")
            except (asyncio.TimeoutError, OSError, UnicodeDecodeError) as e:
                results.append(f"$ {cmd}\nError: {e}")

        summary = data.get("summary", "Task completed")
        if results:
            summary += "\n\nCommand output:\n" + "\n".join(results)
        
        return summary
