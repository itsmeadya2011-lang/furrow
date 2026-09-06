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
        files = list(self.task.files) if self.task.files else self.client.list_files(self.client.settings.workspace)

        file_context: list[str] = []
        for f in files:
            try:
                content = await self.client.read_file(f)
                file_context.append(f"--- {f} ---\n{content}\n")
            except FileNotFoundError:
                file_context.append(f"--- {f} ---\n[file does not exist yet]\n")

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(files)}\n\n"
            + "\n".join(file_context)
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            parsed = json.loads(response)
            edits = parsed.get("edits", [])
        except (json.JSONDecodeError, TypeError):
            return response

        results: list[str] = []
        for edit in edits:
            path = edit.get("path")
            content = edit.get("content", "")
            if path is None:
                continue
            await self.client.write_file(path, content)
            results.append(f"Updated {path}")

        return "; ".join(results) if results else response
