from __future__ import annotations

import json

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self) -> str:
        # Read relevant files first to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:  # noqa: BLE001
                file_contents[file_path] = f"[Could not read {file_path}]"
        
        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant file contents:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content[:2000]}\n"
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse structured response for file operations
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                # Execute file operations
                for file_op in data["files"]:
                    if file_op.get("action") == "write":
                        await self.client.write_file(file_op["path"], file_op["content"])
                    elif file_op.get("action") == "read":
                        # Already read above
                        pass
                return data.get("summary", "Task completed with file operations")
        except (json.JSONDecodeError, ValueError):
            pass
        
        return response
