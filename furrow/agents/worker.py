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
        # Read context files
        context = ""
        if self.task.files:
            for file_path in self.task.files:
                try:
                    content = await self.client.read_file(file_path)
                    context += f"\n--- {file_path} ---\n{content}\n"
                except FileNotFoundError:
                    context += f"\n--- {file_path} ---\n(File not found, will create)\n"
        
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Current file contents:
{context}

Return a JSON object with your changes:
{{
  "changes": [
    {{"file": "path/to/file.py", "content": "new file content"}}
  ],
  "summary": "What you changed"
}}"""
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")
            
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return raw response if parsing fails
            return f"Worker completed (raw response): {response[:500]}"
