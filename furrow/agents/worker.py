from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        log.info("worker.task.start", task_id=self.task.id, description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        raw = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("worker.task.json_error", task_id=self.task.id, error=str(exc))
            raise ValueError(f"Worker LLM response was not valid JSON: {exc}\nRaw response:\n{raw}") from exc

        if not isinstance(parsed, dict):
            log.error("worker.task.shape_error", task_id=self.task.id, type=type(parsed).__name__)
            raise ValueError(f"Worker LLM response JSON must be an object, got {type(parsed).__name__}\nRaw response:\n{raw}")

        files = parsed.get("files")
        if not isinstance(files, list) or not files:
            log.error("worker.task.no_files", task_id=self.task.id)
            raise ValueError(f"Worker LLM response must contain a non-empty 'files' array\nRaw response:\n{raw}")

        summary = parsed.get("summary", "")
        workspace = self.client.settings.workspace
        written: list[str] = []

        for entry in files:
            if not isinstance(entry, dict):
                log.error("worker.task.entry_shape", task_id=self.task.id, type=type(entry).__name__)
                raise ValueError(f"Each file entry must be an object, got {type(entry).__name__}")
            path = entry.get("path")
            content = entry.get("content")
            if not path or not isinstance(path, str):
                log.error("worker.task.missing_path", task_id=self.task.id, entry=entry)
                raise ValueError(f"File entry is missing a non-empty 'path': {entry!r}")
            if not isinstance(content, str) or not content:
                log.error("worker.task.missing_content", task_id=self.task.id, path=path)
                raise ValueError(f"File entry for '{path}' is missing non-empty 'content'")

            resolved = (workspace / path).resolve()
            await self.client.write_file(resolved, content)
            written.append(path)

        log.info("worker.task.files_written", task_id=self.task.id, count=len(written), files=written)
        files_list = ", ".join(written) if written else "none"
        result = f"Summary: {summary}\nFiles written: {files_list}"
        self.task.result = result
        return result
