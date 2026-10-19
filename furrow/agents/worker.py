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
        logger.info("task_started", task_id=self.task.id)

        # Read existing files
        file_contents = {}
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
                file_contents[path] = content
            except FileNotFoundError:
                file_contents[path] = "(file does not exist yet)"

        files_str = "\n\n---\n\n".join(
            f"File: {path}\n```\n{content}\n```"
            for path, content in file_contents.items()
        ) or "No files to modify."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nCurrent files:\n{files_str}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            written = []
            for path, content in data.get("files", {}).items():
                await self.client.write_file(path, content)
                written.append(path)
            logger.info("task_completed", task_id=self.task.id, files_written=written)
            return f"Wrote {len(written)} file(s): {', '.join(written)}"
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("task_parse_failed", task_id=self.task.id, error=str(e), response=response[:200])
            return f"Failed to parse LLM response as JSON: {e}"
