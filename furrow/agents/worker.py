from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger("furrow.worker")


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
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Worker returned non-JSON response for task %s: %s", self.task.id, response[:200])
            return response.strip()

        summary = data.get("summary", "No summary provided.")
        files = data.get("files", [])

        if not isinstance(files, list):
            logger.warning("Worker 'files' field is not a list for task %s", self.task.id)
            files = []

        for file_entry in files:
            path = file_entry.get("path")
            content = file_entry.get("content")
            if not path or content is None:
                continue
            await self.client.write_file(path, content)
            logger.info("Wrote %s for task %s", path, self.task.id)

        if files:
            summary += f" Wrote {len(files)} file(s)."
        return summary
