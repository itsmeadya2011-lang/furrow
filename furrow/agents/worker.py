from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


FILE_BLOCK_RE = re.compile(
    r"---FILE:\s*(?P<path>[^\n]+?)\s*---\s*(?P<content>.*?)---END FILE---",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        errors: list[str] = []

        # Gather file contents for context
        file_context = ""
        for f in self.task.files:
            try:
                content = await self.client.read_file(workspace / f)
                file_context += f"\n--- {f} ---\n{content}\n"
            except Exception as exc:
                errors.append(f"Could not read {f}: {exc}")

        # List workspace files
        try:
            files = self.client.list_files(workspace)
            file_list = ", ".join(files) if files else "(empty workspace)"
        except Exception as exc:
            file_list = f"(could not list files: {exc})"
            errors.append(f"Could not list workspace files: {exc}")

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Workspace files: {file_list}\n"
        )
        if file_context:
            prompt += f"\nFile contents:\n{file_context}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Apply any file changes in the response
        written = []
        for match in FILE_BLOCK_RE.finditer(response):
            path = match.group("path").strip()
            content = match.group("content")
            try:
                await self.client.write_file(workspace / path, content)
                written.append(path)
            except Exception as exc:
                errors.append(f"Could not write {path}: {exc}")

        summary = response
        if written:
            summary += f"\n\nWrote files: {', '.join(written)}"
        if errors:
            summary += f"\n\nErrors: {'; '.join(errors)}"
        return summary