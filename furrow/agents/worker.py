from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Runs a single planned task with an LLM and returns its prose summary.

    Args:
        task: The task to implement. Its ``id`` is echoed into the prompt for
            traceability.
        client: Injected LLM client. When omitted a default one is built.
        settings: Configuration used to build the default client. Ignored - with
            a warning - when it conflicts with ``client.settings``; the explicitly
            passed ``client`` always wins, as it did before.
    """

    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        if client is None:
            self.client = LLMClient() if settings is None else LLMClient(settings=settings)
        else:
            client_settings = getattr(client, "settings", None)
            if (
                settings is not None
                and client_settings is not None
                and settings is not client_settings
            ):
                warnings.warn(
                    "WorkerAgent got both client= and settings=; the client's own settings "
                    "take precedence and the settings argument is ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.client = client
        resolved = getattr(self.client, "settings", None)
        if resolved is None:
            resolved = settings if settings is not None else default_settings
        self.settings: Settings = resolved

    async def run(self) -> str:
        """Execute the task and return the worker's prose summary of its changes."""
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = "\n".join(
            [
                WORKER_PROMPT,
                "",
                f"Task id: {self.task.id}",
                f"Task: {self.task.description}",
                f"Files to touch: {files}",
                "",
            ]
        )
        return await self.client.complete(prompt, model=self.settings.worker_model)