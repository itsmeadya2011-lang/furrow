from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger("furrow.worker")


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        log.info("worker_started", task_id=self.task.id, description=self.task.description)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._apply_changes(response)
        log.info("worker_completed", task_id=self.task.id)
        return response

    async def _apply_changes(self, response: str) -> None:
        file_blocks = re.findall(r"FILE:\s*(.+?)\n```[\w]*\n(.*?)```", response, re.DOTALL)
        for rel_path, content in file_blocks:
            target = self.workspace / rel_path.strip()
            target.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(target, content)
            log.info("file_written", path=str(target))
