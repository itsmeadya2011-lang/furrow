from __future__ import annotations

import asyncio
import json
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

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = "<file not found or not readable>"

        # Build prompt with file context
        files_context = ""
        if file_contents:
            files_context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn a JSON object with your changes:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file.py\", \"content\": \"new file content\"}},\n    ...\n  ],\n  \"commands\": [\n    {{\"cmd\": [\"pytest\", \"-q\"], \"description\": \"Run tests\"}},\n    ...\n  ],\n  \"summary\": \"Brief description of what was done\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            commands = data.get("commands", [])
            summary = data.get("summary", "Task completed")
            
            # Apply file changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            
            # Run commands
            for cmd_spec in commands:
                cmd = cmd_spec.get("cmd", [])
                description = cmd_spec.get("description", "")
                if cmd:
                    if description:
                        self.client.console.print(f"[dim]Running: {description}[/dim]") if hasattr(self.client, 'console') else None
                    exit_code, stdout, stderr = await self.client.run_command(cmd)
                    if exit_code != 0:
                        summary += f"\nCommand failed ({description}): {stderr}"
            
            return summary
        except json.JSONDecodeError:
            # Fallback: try to extract summary from response
            return f"Completed (raw response): {response[:200]}"
