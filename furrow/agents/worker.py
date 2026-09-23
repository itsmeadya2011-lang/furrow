from __future__ import annotations

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
        self.workspace = Path(self.client.settings.workspace).resolve()

    async def run(self) -> str:
        """Execute the task by reading relevant files, making changes, and writing results."""
        # First, read any specified files to understand current state
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    file_contents[file_path] = content
                except Exception as e:
                    file_contents[file_path] = f"Error reading: {e}"

        # Build prompt with file contents
        files_context = ""
        if file_contents:
            files_context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn JSON with this shape:\n{{\n  \"actions\": [\n    {{\"type\": \"write\", \"path\": \"src/file.py\", \"content\": \"...\"}},\n    {{\"type\": \"read\", \"path\": \"src/file.py\"}}\n  ],\n  \"summary\": \"Brief description of changes made\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            actions = data.get("actions", [])
            summary = data.get("summary", "Task completed")

            # Execute actions
            results = []
            for action in actions:
                if action.get("type") == "write":
                    path = action.get("path", "")
                    content = action.get("content", "")
                    full_path = self.workspace / path
                    await self.client.write_file(full_path, content)
                    results.append(f"Wrote {path}")
                elif action.get("type") == "read":
                    path = action.get("path", "")
                    full_path = self.workspace / path
                    if full_path.exists():
                        content = await self.client.read_file(full_path)
                        results.append(f"Read {path}: {len(content)} chars")
                    else:
                        results.append(f"File not found: {path}")

            return f"{summary}\nActions: {'; '.join(results)}"
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: just return the raw response
            return f"Worker response (raw): {response[:500]}"