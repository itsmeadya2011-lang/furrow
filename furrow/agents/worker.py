from __future__ import annotations

import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files = self.task.files or []
        project_structure = self._format_project_structure()
        prompt = (
            f"{WORKER_PROMPT.format(project_structure=project_structure)}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(files) if files else 'any'}\n"
        )
        logger.info("worker_started", task_id=self.task.id, description=self.task.description)
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info("worker_completed", task_id=self.task.id)
        return result

    def _format_project_structure(self) -> str:
        try:
            files = self.client.list_files(self.client.settings.workspace)
            if not files:
                return "(empty workspace)"
            return "\n".join(files[:50])
        except Exception:
            return "(could not list workspace files)"
