from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or default_settings
        self.workspace = Path(workspace or self.settings.workspace)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        data = self.client.extract_json(response)

        summary = data.get("summary", "No summary provided.")
        edits = data.get("edits", []) or []

        for edit in edits:
            path = edit.get("path")
            content = edit.get("content", "")
            if not path:
                continue
            resolved = (self.workspace / path).resolve()
            if self.workspace.resolve() not in resolved.parents and resolved != self.workspace.resolve():
                raise ValueError(f"Refusing to edit file outside workspace: {path}")
            await self.client.write_file(resolved, content)

        return summary
