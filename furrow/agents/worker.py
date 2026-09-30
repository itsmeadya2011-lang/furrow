from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, WorkerResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        existing = [f for f in self.task.files if (workspace / f).exists()]
        file_context = ""
        if existing:
            parts = ["Existing file context:"]
            for f in existing:
                content = await self.client.read_file(workspace / f)
                parts.append(f"\n--- {f} ---\n{content}")
            file_context = "\n".join(parts) + "\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Workspace: {workspace}\n"
            f"{file_context}"
        )

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
            result = WorkerResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            return f"Task {self.task.id}: raw response (JSON parse failed: {e}): {response.strip()[:500]}"

        files_changed: list[str] = []
        for edit in result.edits:
            target = Path(edit.path)
            if not target.is_absolute():
                target = workspace / edit.path
            await self.client.write_file(target, edit.content)
            files_changed.append(edit.path)

        files_str = ", ".join(files_changed) if files_changed else "none"
        return f"Task {self.task.id}: {result.summary} (files: {files_str})"
