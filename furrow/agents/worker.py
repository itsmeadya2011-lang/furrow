from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import TaskModel, TaskStatus
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

_MAX_PRIOR_ERROR_CHARS = 2000


class WorkerAgent:
    """Executes one task and keeps that task's status and attempt count honest."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            resolved = settings
        elif client is not None:
            resolved = client.settings
        else:
            resolved = default_settings
        self.task = task
        self.client = client or LLMClient(settings=resolved)
        self.settings: Settings = resolved

    async def run(self) -> str:
        """Execute the task once, updating ``status``/``attempts``/``error``.

        Raises on LLM failure rather than returning an empty string, so the
        orchestrator sees a failure instead of an empty success. A retry picks up
        the previous error text and is told to address it.
        """
        previous_error = self.task.error if self.task.attempts > 0 else None
        self.task.attempts += 1
        self.task.status = TaskStatus.RUNNING
        self.task.error = None

        prompt = self._build_prompt(previous_error)
        try:
            response = await self.client.complete(
                prompt,
                system=WORKER_SYSTEM_PROMPT,
                model=self.settings.worker_model,
            )
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

        if not response or not response.strip():
            message = "LLM returned an empty response"
            self._fail(message)
            raise RuntimeError(f"Task {self.task.id} failed: {message}")

        self.task.status = TaskStatus.COMPLETED
        self.task.result = response.strip()
        return response.strip()

    def _fail(self, message: str) -> None:
        logger.warning("task %s attempt %d failed: %s", self.task.id, self.task.attempts, message)
        self.task.status = TaskStatus.FAILED
        self.task.error = message

    def _build_prompt(self, previous_error: str | None) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any file you judge necessary"
        parts = [
            WORKER_PROMPT,
            f"Task {self.task.id}: {self.task.description}",
            f"Files you may touch (do not edit anything else): {files}",
        ]
        if previous_error:
            parts.append(
                f"Your previous attempt at this task failed:\n{previous_error[:_MAX_PRIOR_ERROR_CHARS]}\n"
                "Diagnose why it failed and fix that cause this time. Do not repeat the same "
                "approach without changing something."
            )
        return "\n\n".join(parts)
