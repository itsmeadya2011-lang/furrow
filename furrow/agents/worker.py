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

    async def run(self) -> dict:
        context = ""
        for file_path in self.task.files:
            full_path = Path(self.client.settings.workspace) / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = await self.client.read_file(full_path)
                    context += f"\n\n--- {file_path} ---\n{content}"
                except Exception:
                    pass

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
{context}
"""
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            files_modified = []
            for file_entry in data.get("files_modified", []):
                path = file_entry.get("path")
                content = file_entry.get("content", "")
                if path:
                    await self.client.write_file(Path(self.client.settings.workspace) / path, content)
                    files_modified.append(path)
            return {
                "files_modified": files_modified,
                "summary": data.get("summary", ""),
                "success": data.get("success", False),
            }
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "files_modified": [],
                "summary": f"Failed to parse worker response: {e}",
                "success": False,
            }
