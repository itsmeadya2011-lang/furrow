from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow import logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = logger.bind(component="worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found or error reading: {file_path}]"

        files_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        ) if file_contents else "No existing files to reference."

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting file contents:\n{files_context}\n\nReturn a JSON object with this shape:\n{{\n  \"changes\": [\n    {{\"file\": \"path/to/file\", \"content\": \"new file content\"}}\n  ],\n  \"summary\": \"What was changed\"\n}}"
        log.debug("worker_start", task_id=self.task.id, description=self.task.description[:100])
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply the changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
            
            log.info("worker_completed", task_id=self.task.id, changes=len(changes), summary=summary)
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            log.error("worker_parse_failed", task_id=self.task.id, error=str(e), response=response[:200])
            # Fallback: just return the raw response
            return f"Worker response (parse error: {e}): {response}"
