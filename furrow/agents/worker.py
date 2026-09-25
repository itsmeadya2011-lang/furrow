from __future__ import annotations

import json
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

    async def _gather_file_context(self) -> str:
        blocks = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
            except FileNotFoundError:
                continue
            blocks.append(f"### {path}\n```\n{content}\n```\n")
        return "\n".join(blocks)

    async def _apply_changes(self, changes: list[dict]) -> None:
        for change in changes:
            action = change.get("action")
            file = change.get("file")
            content = change.get("content", "")
            if action in ("create", "edit"):
                await self.client.write_file(file, content)
            elif action == "delete":
                try:
                    os.remove(file)
                except OSError:
                    pass

    async def run(self) -> str:
        context = await self._gather_file_context()
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"Existing file context:\n{context}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            parsed = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return response
        if isinstance(parsed, dict) and "changes" in parsed:
            changes = parsed.get("changes") or []
            await self._apply_changes(changes)
            return parsed.get("summary", response)
        return response
