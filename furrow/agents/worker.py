from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class FileChange:
    def __init__(self, path: str, content: str) -> None:
        self.path = path
        self.content = content


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_to_touch = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_to_touch}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: treat the whole response as a summary with no file changes
            return f"No file changes applied. LLM response: {response}"

        changes: list[FileChange] = []
        for entry in data.get("files", []):
            path = entry.get("path")
            content = entry.get("content", "")
            if path:
                changes.append(FileChange(path=path, content=content))

        for change in changes:
            await self.client.write_file(change.path, change.content)

        summary = data.get("summary", "Files written successfully.")
        issues = data.get("issues", [])
        if issues:
            summary += f"\nIssues: {'; '.join(issues)}"

        return summary
