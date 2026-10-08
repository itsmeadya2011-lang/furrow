from __future__ import annotations

import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # First, read the files mentioned in the task to understand context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.client.settings.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        # Build context from files
        context = ""
        if file_contents:
            context = "\n\nRelevant file contents:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to parse the response for file operations
        return await self._execute_response(response)

    async def _execute_response(self, response: str) -> str:
        """Parse and execute file operations from the LLM response."""
        # The worker should return JSON with file operations
        # Expected format: {"files": [{"path": "...", "content": "..."}, ...], "summary": "..."}
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                results = []
                for file_op in data.get("files", []):
                    path = file_op.get("path")
                    content = file_op.get("content")
                    if path and content is not None:
                        await self.client.write_file(self.client.settings.workspace / path, content)
                        results.append(f"Written: {path}")
                summary = data.get("summary", "Files updated")
                return f"{summary}\n" + "\n".join(results)
        except json.JSONDecodeError:
            pass

        # Fallback: return the response as-is (for backward compatibility)
        return response