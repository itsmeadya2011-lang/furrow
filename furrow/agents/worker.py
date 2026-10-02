from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["WorkerAgent"]

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files: dict[str, str] = {}
        for f in self.task.files:
            try:
                files[f] = await self.client.read_file(f)
            except (FileNotFoundError, OSError):
                log.debug("file_not_found", file=f)

        context_parts = [f"--- {f} ---\n{content}\n" for f, content in files.items()]
        context = "\n".join(context_parts)

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Current file contents:\n{context}"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        changes, summary = self._parse_response(response)

        for change in changes:
            file_path = change["file"]
            content = change["content"]
            log.info("writing_file", file=file_path)
            await self.client.write_file(file_path, content)

        return summary

    def _parse_response(self, response: str) -> tuple[list[dict], str]:
        cleaned = response.strip()
        fence = re.search(r"```(?:[a-zA-Z]+)?\n(.*)```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                data = None
            if isinstance(data, dict) and "changes" in data and "summary" in data:
                return data["changes"], data["summary"]
        return [], response
