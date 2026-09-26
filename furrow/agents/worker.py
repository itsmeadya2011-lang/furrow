from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        file_context = await self._read_context_files()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task description: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"\nFile context:\n{file_context}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("json_parse_failed", error=str(e), response=response)
            return f"Error: Failed to parse LLM response as JSON ({e})\nRaw response:\n{response}"
        summary = data.get("summary", "No summary provided")
        files = data.get("files", {})
        written = await self._write_files(files)
        written_list = ", ".join(written) if written else "none"
        return f"{summary}\n\nFiles written: {written_list}"

    async def _read_context_files(self) -> str:
        blocks = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
                blocks.append(f"--- {path} ---\n{content}")
                logger.debug("context_file_read", path=path)
            except FileNotFoundError:
                blocks.append(f"--- {path} ---\n(new file)")
                logger.debug("context_file_missing", path=path)
            except OSError as e:
                blocks.append(f"--- {path} ---\n(error reading file: {e})")
                logger.warning("context_file_read_error", path=path, error=str(e))
        return "\n\n".join(blocks) if blocks else "(no files specified)"

    async def _write_files(self, files: dict[str, str]) -> list[str]:
        written: list[str] = []
        for file_path, content in files.items():
            try:
                await self.client.write_file(file_path, content)
                written.append(file_path)
                logger.info("file_written", path=file_path)
            except OSError as e:
                logger.error("file_write_failed", path=file_path, error=str(e))
        return written
