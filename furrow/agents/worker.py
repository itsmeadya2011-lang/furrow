from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, WorkerResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_str}\n"
        logger.info("worker.start", task_id=self.task.id, task=self.task.description)

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            result = WorkerResult(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("worker.parse_failed", task_id=self.task.id, error=str(e))
            return f"[PARSE_ERROR] Could not parse structured output. Raw response:\n{response}"

        # Write files to disk
        written: list[str] = []
        for file_edit in result.files:
            await self.client.write_file(file_edit.path, file_edit.content)
            written.append(file_edit.path)

        summary = result.summary or "(no summary)"
        if written:
            summary = f"Files written: {', '.join(written)}\n{summary}"

        logger.info("worker.done", task_id=self.task.id, files_written=len(written))
        return summary
