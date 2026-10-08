from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import logger

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
        except json.JSONDecodeError:
            logger.info("Task completed", task_id=self.task.id)
            return f"Failed to parse JSON response. Raw response:\n{response}"

        if not changes:
            logger.info("Task completed", task_id=self.task.id)
            return "No changes made."

        summaries = []
        for change in changes:
            file_path = change.get("file")
            action = change.get("action")
            content = change.get("content")

            if not file_path or not action or content is None:
                continue

            await self.client.write_file(file_path, content)
            summaries.append(f"{action.capitalize()}d {file_path}")

        logger.info("Task completed", task_id=self.task.id)
        return "Changes applied:\n" + "\n".join(summaries)
