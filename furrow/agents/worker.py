from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_JSON_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL | re.IGNORECASE)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'none specified'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._apply_changes(response)

    def _strip_json_fence(self, text: str) -> str:
        match = _JSON_FENCE.match(text.strip())
        if match:
            return match.group(1)
        return text

    async def _apply_changes(self, response: str) -> str:
        """Parse the LLM response for file changes and write them to disk.

        The worker is expected to emit JSON of the shape:
            {
                "changes": [{"file": "path/to/file.py", "content": "..."}],
                "summary": "brief summary of what changed"
            }
        If no parseable changes are found, the raw response is returned unchanged
        so callers still surface whatever the model produced.
        """
        cleaned = self._strip_json_fence(response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return response

        if not isinstance(data, dict):
            return response

        changes = data.get("changes", [])
        written: list[str] = []
        if isinstance(changes, list):
            for change in changes:
                if not isinstance(change, dict):
                    continue
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)
                    written.append(str(file_path))

        summary = data.get("summary", "") or ""
        if written:
            return (f"{summary}\n" if summary else "") + f"Wrote files: {', '.join(written)}"
        return summary or response
