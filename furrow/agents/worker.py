from __future__ import annotations

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
        # Read existing files if specified
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""  # New file

        # Build prompt with file context
        files_context = ""
        if file_contents:
            files_context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to modify: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn JSON with this shape:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"full file content after changes\"\n    }}\n  ],\n  \"summary\": \"Brief summary of changes made\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            import json
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)

            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: just return the response as summary
            return f"Worker response (parse error: {e}): {response[:500]}"