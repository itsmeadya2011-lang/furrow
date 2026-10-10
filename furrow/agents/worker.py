from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()

    async def run(self) -> str:
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            full_path = self.settings.workspace / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = await self.client.read_file(full_path)
                except Exception:
                    file_contents[file_path] = "<error reading file>"

        context = ""
        if file_contents:
            context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}"
        
        logger.debug("worker_start", task_id=self.task.id, description=self.task.description)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse structured response for file writes
        # Expected format: JSON with "files" array of {path, content} and "summary"
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                for file_data in data["files"]:
                    if "path" in file_data and "content" in file_data:
                        full_path = self.settings.workspace / file_data["path"]
                        await self.client.write_file(full_path, file_data["content"])
                        logger.info("file_written", path=file_data["path"])
                logger.info("worker_completed", task_id=self.task.id, summary=data.get("summary", ""))
                return data.get("summary", "Files written successfully")
        except (json.JSONDecodeError, ValueError):
            pass
        
        logger.info("worker_completed", task_id=self.task.id)
        return response
