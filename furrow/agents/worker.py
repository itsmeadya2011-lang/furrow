from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.jsonparse import extract_json
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileEdit, TaskModel, WorkerResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_FILE_SNIPPET_CHARS = 8000


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.settings = settings if settings is not None else default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> WorkerResult:
        prompt = await self._build_prompt()
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(extract_json(response))
            return WorkerResult(
                summary=str(data.get("summary", "")),
                edits=[
                    FileEdit(path=str(e.get("path", "")), content=str(e.get("content", "")))
                    for e in data.get("edits", [])
                    if isinstance(e, dict) and e.get("path")
                ],
            )
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError) as e:
            return WorkerResult(
                summary=f"Worker output could not be parsed as JSON ({e}): {response[:500]}",
                edits=[],
            )

    async def _build_prompt(self) -> str:
        parts = [
            WORKER_PROMPT,
            "",
            f"Task: {self.task.description}",
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}",
        ]
        sections: list[str] = []
        for rel in self.task.files:
            try:
                content = await self.client.read_file(self.settings.workspace / rel)
            except (OSError, ValueError):
                continue
            if len(content) > MAX_FILE_SNIPPET_CHARS:
                content = content[:MAX_FILE_SNIPPET_CHARS] + "\n... (truncated)"
            sections.append(f"--- {rel} ---\n{content}")
        if sections:
            parts += ["", "Current file contents:"] + sections
        return "\n".join(parts) + "\n"