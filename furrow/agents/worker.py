from __future__ import annotations

from typing import TYPE_CHECKING

from structlog import get_logger

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.files_changed: list[str] = []

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        result = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.info("worker_completed", task_id=self.task.id, files=self.files_changed)
        return result

    def parse_file_changes(self, result: str) -> list[str]:
        """Extract file paths mentioned in a worker result string."""
        import re
        paths = re.findall(r'`([^`]+)`', result)
        self.files_changed = [p for p in paths if p.strip()]
        return self.files_changed

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.id,
            "description": self.task.description,
            "status": self.task.status,
            "result": self.task.result,
            "files_changed": self.files_changed,
        }
