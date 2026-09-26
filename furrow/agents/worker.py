from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.util import parse_json_response

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._apply(response)

    async def _apply(self, response: str) -> str:
        try:
            data = parse_json_response(response)
        except (json.JSONDecodeError, ValueError):
            # Fallback: the LLM did not return structured output. Return the raw
            # response as the summary without writing any files.
            return response.strip()

        summary = data.get("summary") or response.strip()
        files = data.get("files") or []

        for entry in files:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not path:
                continue
            action = entry.get("action", "modify")
            if action == "delete":
                p = Path(path)
                if p.exists():
                    p.unlink()
                continue
            content = entry.get("content", "")
            await self.client.write_file(path, content)

        return summary
