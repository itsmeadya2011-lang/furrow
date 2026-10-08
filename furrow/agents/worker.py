from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class FileChange(BaseModel):
    path: str
    content: str
    action: str  # "create", "update", "delete"


class WorkerResult(BaseModel):
    summary: str
    changes: list[FileChange] = Field(default_factory=list)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read existing files for context
        file_contents = {}
        for file_path in self.task.files:
            full_path = Path(file_path)
            if full_path.exists():
                file_contents[file_path] = await self.client.read_file(full_path)
        
        # Build context from existing files
        context = ""
        if file_contents:
            context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}\n\nReturn JSON with this shape:\n{{\n  \"summary\": \"Brief summary of changes\",\n  \"changes\": [\n    {{\"path\": \"file.py\", \"content\": \"full file content\", \"action\": \"create|update|delete\"}}\n  ]\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            result = WorkerResult(**data)
            
            # Apply changes
            for change in result.changes:
                if change.action == "delete":
                    Path(change.path).unlink(missing_ok=True)
                else:
                    await self.client.write_file(change.path, change.content)
            
            return result.summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return raw response as summary
            return f"Failed to parse structured output: {e}\nRaw response: {response}"
