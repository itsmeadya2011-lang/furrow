from __future__ import annotations

import os
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

    async def run(self, project_context: str = "") -> str:
        workspace = self.client.settings.workspace
        file_listing = []
        for root, dirs, files in os.walk(str(workspace)):
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                file_path = os.path.relpath(os.path.join(root, f), str(workspace))
                file_listing.append(file_path)
        file_listing_text = "\n".join(file_listing[:200])
        if len(file_listing) > 200:
            file_listing_text += f"\n... and {len(file_listing) - 200} more files"

        prompt = f"""{WORKER_PROMPT.format(project_context=project_context)}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Project files:
{file_listing_text}
"""
        return await self.client.complete(prompt, model=self.client.settings.worker_model)