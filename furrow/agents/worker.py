from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
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
        # First, read any relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                logger.warning("failed_to_read_file", file=file_path, error=str(e))

        # Build prompt with file context
        files_context = ""
        if file_contents:
            files_context = "\n\n### File Context:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n\nReturn a JSON object with your changes:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"What you changed\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse and apply changes
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Completed task")

            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
                    logger.info("file_written", file=file_path)

            return summary
        except json.JSONDecodeError:
            # Fallback: return raw response if not valid JSON
            return response
