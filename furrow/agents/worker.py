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
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> WorkerResult:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            result = WorkerResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = WorkerResult(
                summary=response[:500],
                files_changed=[],
                edits=[],
            )

        await self._apply_edits(result)
        return result

    async def _apply_edits(self, result: WorkerResult) -> None:
        for edit in result.edits:
            if edit.new_content is None:
                p = Path(edit.path)
                if p.exists():
                    p.unlink()
            elif edit.old_content is None:
                await self.client.write_file(edit.path, edit.new_content)
            else:
                existing = await self.client.read_file(edit.path)
                updated = existing.replace(edit.old_content, edit.new_content, 1)
                if updated == existing:
                    # No match found; write full new content as fallback
                    await self.client.write_file(edit.path, edit.new_content)
                else:
                    await self.client.write_file(edit.path, updated)
