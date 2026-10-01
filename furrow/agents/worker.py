from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

# Regex to match FILE blocks:
#   {FILE: path/to/file.py}
#   ```language
#   ...content...
#   ```
#   {END_FILE}
_FILE_BLOCK_RE = re.compile(
    r"\{FILE:\s*(?P<path>[^\n}\s]+)\}\s*```(?:\w+)?\n(?P<content>.*?)```",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = WORKER_PROMPT.format(
            description=self.task.description,
            files=files,
            goal=getattr(self.client.settings, "goal", ""),
        )
        logger.info("worker_run", task_id=self.task.id, description=self.task.description)
        response = await self.client.complete(
            prompt, system=WORKER_SYSTEM, model=self.client.settings.worker_model
        )

        # Parse and write any files the LLM produced
        files_written = await self._parse_and_write_files(response)
        if files_written:
            logger.info("worker_wrote_files", task_id=self.task.id, files=files_written)

        return self._build_summary(response, files_written)

    async def _parse_and_write_files(self, output: str) -> list[str]:
        """Parse LLM output for FILE blocks and write them to disk."""
        written: list[str] = []
        for match in _FILE_BLOCK_RE.finditer(output):
            raw_path = match.group("path").strip()
            content = match.group("content")
            path = self.client.resolve_path(raw_path)
            try:
                await self.client.write_file(path, content)
                written.append(raw_path)
            except OSError as e:
                logger.error("worker_write_failed", path=raw_path, error=str(e))
        return written

    def _build_summary(self, response: str, files_written: list[str]) -> str:
        """Build a concise summary, stripping FILE blocks from the response."""
        clean = _FILE_BLOCK_RE.sub("", response).strip()
        parts = [f"Files written: {', '.join(files_written)}"] if files_written else []
        if clean:
            parts.append(clean)
        return "\n\n".join(parts) if parts else "Task completed."
