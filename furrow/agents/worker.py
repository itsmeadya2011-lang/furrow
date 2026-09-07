from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or (LLMClient(settings=settings) if settings is not None else LLMClient())

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._apply_result(response)

    async def _apply_result(self, response: str) -> str:
        """Parse the LLM response and write any declared files to disk."""
        data = self._parse_json(response)
        if data is None:
            log.error("worker.parse_failed", task_id=self.task.id)
            return f"Failed to parse worker result.\nResponse: {response[:200]}"
        if not isinstance(data, dict):
            log.error("worker.parse_failed", task_id=self.task.id, reason="json_not_object")
            return f"Worker result is not a JSON object.\nResponse: {response[:200]}"

        files: list[dict[str, Any]] = data.get("files", [])
        summary: str = data.get("summary", "")

        for file_entry in files:
            path = file_entry.get("path")
            content = file_entry.get("content")
            if not path or content is None:
                continue
            try:
                await self.client.write_file(path, content)
                log.info("worker.file_written", task_id=self.task.id, path=path)
            except OSError as e:
                log.error("worker.write_failed", task_id=self.task.id, path=path, error=str(e))
                summary += f"\nFailed to write {path}: {e}"

        return summary

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Parse JSON from a response, stripping markdown code fences if present."""
        # Strip ```json ... ``` or ``` ... ``` fences.
        cleaned = re.sub(r"^```(?:\w+)?\n(.*)\n```$", r"\1", text.strip(), flags=re.DOTALL)
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            log.debug("worker.json_parse_skipped", raw=text[:200])
            return None
