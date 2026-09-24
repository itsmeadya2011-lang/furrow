from __future__ import annotations

import asyncio
import json
import os
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
        self.settings = settings

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description[:100])
        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        file_context = ""
        if file_contents:
            file_context = "\n\n## Relevant Files\n"
            for path, content in file_contents.items():
                file_context += f"\n### {path}\n```\n{content[:3000]}\n```\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{file_context}\n\nReturn a JSON object with:\n- \"changes\": list of {{ \"file\": \"path\", \"content\": \"new content\" }}\n- \"summary\": \"what was done\"\n- \"complete\": true/false"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)
                    logger.info("file_written", file=file_path, size=len(content))
            summary = data.get("summary", "Task completed")
            logger.info("worker_completed", task_id=self.task.id, summary=summary[:100])
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("worker_response_not_json", task_id=self.task.id, error=str(e))
            # Fallback: just return the response as summary
            return response
