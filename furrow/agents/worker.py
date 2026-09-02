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
        workspace = Path(self.client.settings.workspace)
        existing_files = self.client.list_files(workspace)[:50]
        files_context = "\n".join(f"- {f}" for f in existing_files) if existing_files else "(empty workspace)"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Existing files in workspace:\n{files_context}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return response

        summary = data.get("summary", "")
        written = []
        for file_info in data.get("files", []):
            path = file_info.get("path")
            content = file_info.get("content", "")
            action = file_info.get("action", "overwrite")
            if not path:
                continue
            target = workspace / path
            if action == "create" and target.exists():
                continue
            await self.client.write_file(target, content)
            written.append(str(path))

        if written:
            summary = f"{summary}\n\nWrote files: {', '.join(written)}"

        return summary
