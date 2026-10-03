from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Files larger than this are not worth a place in the prompt.
MAX_EXCERPT_BYTES = 256 * 1024
# Characters of each file actually shown to the worker.
MAX_EXCERPT_CHARS = 2000
TRUNCATION_MARKER = "\n... [truncated]"


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
        prompt = await self._build_prompt()
        return await self.client.complete(prompt, model=self.client.settings.worker_model)

    async def _build_prompt(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        parts = [WORKER_PROMPT, "", f"Task: {self.task.description}", f"Files to touch: {files}"]
        if self.task.dependencies:
            parts.append(f"Depends on these completed tasks: {', '.join(self.task.dependencies)}")
        excerpts = await self._gather_excerpts()
        if excerpts:
            parts.append("")
            parts.append("Current contents of the files assigned to this task:")
            parts.extend(excerpts)
        return "\n".join(parts)

    async def _gather_excerpts(self) -> list[str]:
        """Read a short excerpt of each assigned file, tolerating every failure."""
        if not self.task.files:
            return []
        workspace = Path(self.client.settings.workspace)
        results = await asyncio.gather(
            *(self._read_excerpt(path, workspace) for path in self.task.files),
            return_exceptions=True,
        )
        excerpts: list[str] = []
        for path, result in zip(self.task.files, results, strict=False):
            if isinstance(result, BaseException):
                excerpts.append(f"--- {path} (unavailable: {result}) ---")
            else:
                excerpts.append(f"--- {path} ---\n{result}")
        return excerpts

    async def _read_excerpt(self, path: str, workspace: Path) -> str:
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else workspace / candidate
        if not resolved.is_file():
            return f"(not found: {path})"
        try:
            too_big = resolved.stat().st_size > MAX_EXCERPT_BYTES
        except OSError as exc:
            return f"(unavailable: {exc})"
        if too_big:
            return f"(skipped: larger than {MAX_EXCERPT_BYTES} bytes)"
        text = await self.client.read_file(resolved)
        if len(text) > MAX_EXCERPT_CHARS:
            return text[:MAX_EXCERPT_CHARS] + TRUNCATION_MARKER
        return text
