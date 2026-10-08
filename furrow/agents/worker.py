from __future__ import annotations

import aiofiles
import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Get context for the files we need to work on
        file_context = await self._get_file_context()

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}

File context:
{file_context}

You must return a JSON object (not wrapped in markdown) with this shape:
{{
  "changes": [
    {{
      "file": "path/to/file.py",
      "content": "new file content",
      "operation": "create|update|delete"
    }}
  ],
  "summary": "Brief description of what was done",
  "complete": true
}}

If the task is not complete, set complete to false and explain in summary.
"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")
            complete = data.get("complete", False)

            # Apply the changes
            for change in changes:
                await self._apply_change(change)

            if not complete:
                self.task.status = "pending"
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat response as summary if not valid JSON
            return f"Worker response (parse error: {e}): {response[:200]}"

    async def _get_file_context(self) -> str:
        """Get content of files relevant to the task."""
        if not self.task.files:
            return "No specific files specified."
        
        context_parts = []
        for file_path in self.task.files:
            full_path = Path(self.settings.workspace) / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"=== {file_path} ===\n{content}")
                except Exception as e:
                    context_parts.append(f"=== {file_path} === (error reading: {e})")
            else:
                context_parts.append(f"=== {file_path} === (does not exist)")
        
        return "\n\n".join(context_parts) if context_parts else "No existing files found."

    async def _apply_change(self, change: dict) -> None:
        """Apply a file change."""
        file_path = Path(self.settings.workspace) / change["file"]
        operation = change.get("operation", "update")
        content = change.get("content", "")

        if operation == "delete":
            if file_path.exists():
                file_path.unlink()
        elif operation == "create":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)
        elif operation == "update":
            file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(file_path, "w") as f:
                await f.write(content)