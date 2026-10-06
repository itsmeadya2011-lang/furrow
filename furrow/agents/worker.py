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
        logger.info("worker_started", task_id=self.task.id, description=self.task.description)
        # First, read any relevant files to understand the context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        # Build the prompt with file context
        context = ""
        if file_contents:
            context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n\nReturn JSON with your changes:\n{{\n  \"changes\": [\n    {{\"path\": \"file.py\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"What you changed\"\n}}"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")

            for change in changes:
                path = change.get("path")
                content = change.get("content")
                if path and content is not None:
                    await self.client.write_file(path, content)

            logger.info("worker_completed", task_id=self.task.id, changes=len(changes))
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: return the raw response
            logger.warning("worker_json_parse_failed", task_id=self.task.id, error=str(e))
            return f"Worker completed (raw response): {response[:500]}"
