from __future__ import annotations

import json
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
        # Read all files mentioned in the task
        file_contents = {}
        for file_path in self.task.files:
            path = Path(file_path)
            if path.exists():
                file_contents[file_path] = await self.client.read_file(path)
            else:
                file_contents[file_path] = ""  # New file

        # Build prompt with file contents
        files_info = "\n\n".join(
            f"=== {path} ===\n{content}" if content else f"=== {path} === (new file)"
            for path, content in file_contents.items()
        )
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles:\n{files_info}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                path = Path(change["path"])
                content = change["content"]
                await self.client.write_file(path, content)
            
            return summary
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback: return the raw response as summary
            return f"Completed (raw): {response[:200]}..."
