from __future__ import annotations

from pathlib import Path

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

MAX_FILE_CHARS = 4000
"""Per-file budget for the source excerpts appended to the worker prompt."""

TRUNCATION_MARKER = "\n... [truncated: showing first {shown} of {total} characters]"
"""Appended to any file excerpt that had to be shortened."""


class WorkerAgent:
    """Executes one task and returns the model's summary of the work."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings = settings if settings is not None else self.client.settings

    async def run(self, goal: str = "", context: str = "") -> str:
        """Execute the task once and return the summary string."""
        return await self.run_attempt(1, goal=goal, context=context)

    async def run_attempt(self, attempt: int = 1, goal: str = "", context: str = "") -> str:
        """Execute the task as attempt ``attempt``.

        Raises:
            LLMError: propagated unchanged so the orchestrator can count it against
                its error budget.
        """
        prompt = await self._build_prompt(attempt, goal, context)
        return await self.client.complete(prompt, model=self.settings.worker_model)

    async def _build_prompt(self, attempt: int, goal: str, context: str) -> str:
        sections = [
            WORKER_PROMPT,
            f"\nTask id: {self.task.id}",
            f"Attempt: {max(1, int(attempt))}",
            f"\nTask description:\n{self.task.description}",
            f"\nFiles this task owns: {', '.join(self.task.files) or 'unspecified'}",
            f"Dependencies: {', '.join(self.task.dependencies) or 'none'}",
        ]
        if goal.strip():
            sections.append(f"\nOverall goal:\n{goal.strip()}")
        if context.strip():
            sections.append(f"\nProject context:\n{context.strip()}")
        excerpts = await self._file_excerpts()
        if excerpts:
            sections.append("\nCurrent content of the files this task owns:\n\n" + excerpts)
        return "".join(sections)

    async def _file_excerpts(self) -> str:
        """Render the current content of each existing file in ``task.files``."""
        blocks: list[str] = []
        for name in self.task.files:
            path = self._resolve(name)
            if path is None or not path.is_file():
                continue
            try:
                content = await self.client.read_file(path)
            except OSError:
                continue
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + TRUNCATION_MARKER.format(
                    shown=MAX_FILE_CHARS, total=len(content)
                )
            blocks.append(f"--- {name} ---\n```\n{content}\n```\n")
        return "\n".join(blocks)

    def _resolve(self, name: str) -> Path | None:
        candidate = Path(name)
        if candidate.is_absolute():
            return candidate
        return self.settings.workspace / candidate


__all__ = ["WorkerAgent"]