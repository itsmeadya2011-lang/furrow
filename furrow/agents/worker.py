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

    async def run(self) -> str:
        # Read context files to give worker awareness of existing code
        context_files = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_files[file_path] = content
            except FileNotFoundError:
                context_files[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                context_files[file_path] = f"[Error reading file: {e}]"

        context_str = ""
        if context_files:
            context_str = "\n\n=== EXISTING FILES ===\n"
            for path, content in context_files.items():
                context_str += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context_str}\n"
        
        # Ask worker to return structured response with file changes
        response = await self.client.complete(
            prompt, 
            model=self.client.settings.worker_model,
            system="You are a coding agent. Return ONLY a JSON object with this structure:\n{\n  \"summary\": \"Brief summary of changes\",\n  \"files\": [\n    {\"path\": \"file/path.py\", \"content\": \"full file content\"}\n  ]\n}\nDo not include markdown or explanation."
        )
        
        try:
            data = json.loads(response)
            # Write any files the worker returned
            for file_change in data.get("files", []):
                await self.client.write_file(file_change["path"], file_change["content"])
            return data.get("summary", "Task completed")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: just return the raw response
            return f"Worker response (could not parse as JSON): {response[:500]}"
