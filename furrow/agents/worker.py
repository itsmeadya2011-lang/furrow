from __future__ import annotations

import json
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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return f"Failed to parse LLM response as JSON: {response}"
        changes = data.get("changes", [])
        written: list[str] = []
        for change in changes:
            path = change.get("path")
            content = change.get("content", "")
            if not path:
                continue
            await self.client.write_file(path, content)
            written.append(path)
        summary = data.get("summary", f"Wrote {len(written)} file(s).")
        return f"{summary}\nFiles written: {', '.join(written) if written else 'none'}"
