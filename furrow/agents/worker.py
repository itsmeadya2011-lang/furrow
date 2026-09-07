from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

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
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        raw_response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        parsed = self._extract_json(raw_response)
        if parsed is None:
            return raw_response

        files = parsed.get("files", {})
        summary = parsed.get("summary", "")

        for path, content in files.items():
            await self.client.write_file(path, content)

        if summary:
            return summary
        return f"Wrote {len(files)} file(s): {', '.join(files.keys())}"

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
