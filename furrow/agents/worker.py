from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Read relevant files first to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(self.settings.workspace / file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                file_contents[file_path] = f"[Error reading {file_path}: {e}]"

        file_context = ""
        if file_contents:
            file_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                file_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{file_context}\n\nReturn a JSON object with:\n{{\n  \"changes\": [{{\n    \"file\": \"path/to/file\",\n    \"content\": \"new file content\"\n  }}],\n  \"summary\": \"What was done\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.client.write_file(self.settings.workspace / file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return the raw response as summary
            return f"Worker response (parse error: {e}): {response}"