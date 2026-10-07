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
        # Read context files if specified
        context = ""
        if self.task.files:
            for file_path in self.task.files:
                path = Path(file_path)
                if path.exists():
                    content = await self.client.read_file(path)
                    context += f"\n--- {file_path} ---\n{content}\n"
                else:
                    context += f"\n--- {file_path} ---\n(File does not exist yet)\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\nContext:\n{context}\n"
        
        # Ask LLM to provide file changes
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse and apply file changes
        return await self._apply_changes(response)

    async def _apply_changes(self, response: str) -> str:
        """Parse LLM response for file operations and apply them."""
        try:
            # Try to parse as JSON with file operations
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                changes = []
                for file_op in data["files"]:
                    path = file_op.get("path", "")
                    content = file_op.get("content", "")
                    action = file_op.get("action", "write")
                    
                    if action == "write" or action == "create":
                        await self.client.write_file(path, content)
                        changes.append(f"Created/updated: {path}")
                    elif action == "delete":
                        Path(path).unlink(missing_ok=True)
                        changes.append(f"Deleted: {path}")
                return f"Applied {len(changes)} file changes:\n" + "\n".join(changes)
        except json.JSONDecodeError:
            pass
        
        # Fallback: return the response as-is
        return response
