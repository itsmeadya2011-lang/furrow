from __future__ import annotations

import os
import re
import textwrap
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
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_start", task_id=self.task.id, description=self.task.description)

        context = ""
        for f in self.task.files:
            if os.path.exists(f):
                try:
                    content = await self.client.read_file(f)
                    context += f"\n--- Existing file: {f} ---\n{content}"
                    logger.debug("context_read", file=f)
                except Exception as e:  # noqa: BLE001
                    logger.warning("context_read_failed", file=f, error=str(e))

        files_str = ", ".join(self.task.files) if self.task.files else "any files needed"
        prompt = (
            f"{WORKER_PROMPT}\n"
            f"\nTask: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
            f"Context files:\n{context}\n"
            f"\nImplement the task. Output file changes using "
            f'<FILE path="...">content</FILE> blocks.'
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        edited = await self._write_file_edits(response)

        summary_lines = [f"Edited {len(edited)} file(s):"]
        for path, ok in edited:
            summary_lines.append(f"  - {path}: {'OK' if ok else 'FAILED'}")
        summary_lines.append("")
        summary_lines.append(response)

        result = "\n".join(summary_lines)
        logger.info("worker_done", task_id=self.task.id, files_edited=len(edited))
        return result

    def _parse_file_edits(self, response: str) -> list[tuple[str, str]]:
        pattern = r'<FILE\s+path="([^"]+)">\s*\n(.*?)\n\s*</FILE>'
        blocks = re.findall(pattern, response, re.DOTALL)
        files = []
        for path, raw_content in blocks:
            path = path.strip()
            content = textwrap.dedent(raw_content)
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n", "", content, count=1)
            if content.endswith("```"):
                content = re.sub(r"\n```$", "", content, count=1)
            files.append((path, content))
        return files

    async def _write_file_edits(self, response: str) -> list[tuple[str, bool]]:
        edits = self._parse_file_edits(response)
        results: list[tuple[str, bool]] = []
        for path, content in edits:
            try:
                await self.client.write_file(path, content)
                logger.info("file_written", path=path, chars=len(content))
                results.append((path, True))
            except Exception as e:  # noqa: BLE001
                logger.error("file_write_failed", path=path, error=str(e))
                results.append((path, False))
        return results
