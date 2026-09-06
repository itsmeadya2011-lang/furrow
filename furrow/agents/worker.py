from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | str | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.workspace = Path(workspace) if workspace else (settings.workspace if settings else None)

    async def run(self) -> str:
        workspace_info = ""
        if self.workspace:
            workspace_info = f"\nWorking directory: {self.workspace}\n"
            try:
                files = self.client.list_files(self.workspace)
                if files:
                    workspace_info += f"Current files: {', '.join(files[:30])}\n"
            except Exception:
                pass

        prompt = (
            f"{WORKER_PROMPT}{workspace_info}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        try:
            result = await self.client.complete(prompt, model=self.client.settings.worker_model)
            logger.info("Worker completed task %s", self.task.id)
            return result
        except Exception as e:
            logger.error("Worker failed on task %s: %s", self.task.id, e)
            raise
