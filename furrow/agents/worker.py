from __future__ import annotations

import re

from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.max_rounds = 3

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def run(self) -> str:
        """Execute the task using file operations and return a summary of changes."""
        logger.debug("worker_started", task_id=self.task.id, description=self.task.description)
        # First, read relevant files to understand the context
        context = await self._gather_context()
        
        # Then plan and execute changes
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nRelevant files context:\n{context}\n\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse the response for file operations
        result = await self._execute_changes(response)
        logger.info("worker_completed", task_id=self.task.id)
        return result

    async def _gather_context(self) -> str:
        """Read relevant files to provide context to the worker."""
        if not self.task.files:
            return "No specific files specified. Worker should explore the codebase."
        
        context_parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"=== {file_path} ===\n{content}")
            except FileNotFoundError:
                context_parts.append(f"=== {file_path} === (not found, will be created)")
            except OSError as e:
                context_parts.append(f"=== {file_path} === (error reading: {e})")
        
        return "\n\n".join(context_parts)

    async def _execute_changes(self, response: str) -> str:
        """Parse response for file operations and execute them."""
        # Try to parse structured response with file operations
        changes_made = []
        
        # Look for code blocks with file paths
        pattern = r'```(\w+)?\n(?:#\s*(.+?)\n)?(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for lang, file_path, content in matches:
            file_path = file_path.strip() if file_path else ""
            content = content.strip()
            
            if file_path and content:
                try:
                    await self.client.write_file(file_path, content)
                    changes_made.append(f"Written: {file_path}")
                    logger.debug("file_written", path=file_path)
                except OSError as e:
                    changes_made.append(f"Error writing {file_path}: {e}")
                    logger.error("file_write_failed", path=file_path, error=str(e))
        
        # If no structured changes found, return the response as summary
        if not changes_made:
            logger.debug("no_structured_changes", response_length=len(response))
            return response
        
        return "Changes made:\n" + "\n".join(changes_made) + f"\n\nSummary: {response}"
