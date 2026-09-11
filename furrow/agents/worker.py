from __future__ import annotations

import json
import structlog
from typing import TYPE_CHECKING

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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        logger.debug("worker_prompt", task_id=self.task.id, prompt=prompt[:500])
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        logger.debug("worker_response", task_id=self.task.id, response=response[:500])
        
        # Try to parse JSON response with file operations
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                return await self._execute_file_operations(data["files"])
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: return the response as summary
        logger.info("worker_completed", task_id=self.task.id)
        return response

    async def _execute_file_operations(self, files: list[dict]) -> str:
        """Execute file operations returned by the LLM."""
        results = []
        for file_op in files:
            path = file_op.get("path", "")
            content = file_op.get("content", "")
            operation = file_op.get("operation", "write")
            
            if operation == "write" or operation == "create":
                await self.client.write_file(path, content)
                results.append(f"Created/updated {path}")
                logger.info("file_written", path=path)
            elif operation == "read":
                file_content = await self.client.read_file(path)
                results.append(f"Read {path}: {file_content[:200]}...")
                logger.info("file_read", path=path)
            elif operation == "delete":
                import os
                os.remove(path)
                results.append(f"Deleted {path}")
                logger.info("file_deleted", path=path)
        
        return "\n".join(results)
