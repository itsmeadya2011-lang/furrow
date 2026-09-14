from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("task.received", task_id=self.task.id, description=self.task.description, files=self.task.files)
        file_contents = {}
        for file_path in self.task.files or []:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
                logger.debug("file.read", path=file_path, size=len(content))
            except FileNotFoundError:
                file_contents[file_path] = ""
                logger.debug("file.not_found", path=file_path)

        files_section = ""
        if file_contents:
            files_section = "\n\nCurrent file contents:\n"
            for path, content in file_contents.items():
                files_section += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_section}"
        logger.debug("llm.call.started", task_id=self.task.id, model=self.client.settings.worker_model)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.debug("llm.call.completed", task_id=self.task.id, response_length=len(response))

        try:
            result = json.loads(response)
            files_to_write = result.get("files", {})
            summary = result.get("summary", "Task completed")

            for path, content in files_to_write.items():
                await self.client.write_file(path, content)
                logger.info("file.written", path=path, size=len(content))

            logger.info("task.completed", task_id=self.task.id, summary=summary, files_written=list(files_to_write.keys()))
            return summary
        except json.JSONDecodeError:
            logger.warning("llm.response.invalid_json", task_id=self.task.id)
            return response
