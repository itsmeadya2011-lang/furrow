from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("task.start", task_id=self.task.id, description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            match = re.search(r'```(?:json)?\n?(.*?)```', response, re.DOTALL)
            json_text = match.group(1) if match else response
            data = json.loads(json_text)
            for file in data.get("files", []):
                logger.info("file.write", path=file["path"], task_id=self.task.id)
                await self.client.write_file(file["path"], file["content"])
            logger.info("task.complete", task_id=self.task.id)
            return data.get("summary", response)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("task.error", task_id=self.task.id, error=str(e))
            return response
