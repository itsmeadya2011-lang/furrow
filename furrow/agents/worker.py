from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self._settings = settings or default_settings

    async def run(self) -> str:
        logger.info("worker.start", task_id=self.task.id, description=self.task.description[:100])
        # Read context files first
        context = await self._read_context_files()
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nContext files:\n{context}\n"
        worker_model = getattr(self.client, 'settings', self._settings).worker_model
        response = await self.client.complete(prompt, model=worker_model)
        
        # Parse response for file operations
        result = await self._execute_file_operations(response)
        logger.info("worker.completed", task_id=self.task.id)
        return result

    async def _read_context_files(self) -> str:
        """Read files listed in task.files to provide context."""
        if not self.task.files:
            return "No specific files provided."
        
        context_parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"=== {file_path} ===\n{content}")
            except OSError as e:
                logger.warning("worker.context_read_failed", file=file_path, error=str(e))
                context_parts.append(f"=== {file_path} === (error reading: {e})")
        return "\n\n".join(context_parts)

    async def _execute_file_operations(self, response: str) -> str:
        """Parse LLM response for file operations and execute them."""
        summary_parts = []
        
        # Try to parse JSON response with file operations
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "files" in data:
                for file_op in data["files"]:
                    path = file_op.get("path", "")
                    content = file_op.get("content", "")
                    operation = file_op.get("operation", "write")
                    
                    if operation == "write" and path:
                        await self.client.write_file(path, content)
                        summary_parts.append(f"Created/updated: {path}")
                        logger.info("worker.file_written", path=path)
                    elif operation == "delete" and path:
                        # Note: LLMClient doesn't have delete, would need to add
                        summary_parts.append(f"Would delete: {path} (not implemented)")
                
                if summary_parts:
                    return "File operations completed:\n" + "\n".join(summary_parts)
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug("worker.response_not_json", error=str(e))
        
        # Fallback: return the raw response as summary
        return response
