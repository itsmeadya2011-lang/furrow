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

    async def _apply_edit(self, edit: dict) -> str:
        path = edit.get("path", "")
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")

        target = Path(path)

        if not old_text:
            await self.client.write_file(target, new_text)
            return f"wrote {path}"

        try:
            current = await self.client.read_file(target)
        except Exception:
            return f"skipped {path}: file not found or unreadable"

        if old_text not in current:
            return f"skipped {path}: old_text not found"

        updated = current.replace(old_text, new_text)
        await self.client.write_file(target, updated)
        return f"edited {path}"

    async def run(self) -> str:
        files_list = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n"
        )
        raw = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw

        if not isinstance(data, dict):
            return raw

        edits = data.get("edits", [])
        summary = data.get("summary", "")
        applied: list[str] = []
        skipped: list[str] = []

        for edit in edits:
            if not isinstance(edit, dict):
                continue
            path = edit.get("path", "")
            if self.task.files and path not in self.task.files:
                skipped.append(path)
                continue
            result = await self._apply_edit(edit)
            applied.append(result)

        parts: list[str] = []
        if summary:
            parts.append(summary)
        if applied:
            parts.append(f"Applied {len(applied)} edit(s): {', '.join(applied)}")
        if skipped:
            parts.append(f"Skipped {len(skipped)} edit(s) outside allowed files: {', '.join(skipped)}")

        return " | ".join(parts) if parts else "No edits applied"
