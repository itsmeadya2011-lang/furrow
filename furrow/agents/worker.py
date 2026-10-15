from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=config or settings)
        self.workspace = (config or settings).workspace

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.workspace / file_path
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)

        files_context = ""
        if file_contents:
            files_context = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn JSON with your changes:\n{{\n  \"changes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"full new file content\"\n    }}\n  ],\n  \"summary\": \"Brief summary of changes\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    full_path = self.workspace / file_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    await self.client.write_file(full_path, content)

            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return raw response as summary
            return f"Worker response (parse error: {e}): {response[:500]}"
