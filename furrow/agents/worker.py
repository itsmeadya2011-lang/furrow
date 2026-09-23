from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

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
        # First, read any relevant files to understand the codebase
        context = await self._gather_context()
        
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Current context:
{context}

Return a JSON object with your changes:
{{
  "changes": [
    {{"file": "path/to/file.py", "content": "new file content"}},
    {{"file": "path/to/new_file.py", "content": "new file content"}}
  ],
  "summary": "Brief summary of changes made"
}}

If no changes needed, return {{"changes": [], "summary": "No changes needed"}}
"""
        response = await self._execute_with_retry(prompt)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)
                    logger.info("file_written", file=file_path, task_id=self.task.id)
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("worker_parse_error", task_id=self.task.id, error=str(e), response=response[:500])
            # Fallback: just return the response as summary
            return response[:500]

    async def _gather_context(self) -> str:
        """Read relevant files to provide context to the worker."""
        context_parts = []
        workspace = Path.cwd()
        
        # Read specified files
        for file_path in self.task.files:
            full_path = workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    context_parts.append(f"=== {file_path} ===\n{content}")
                except Exception as e:
                    logger.warning("failed_to_read_file", file=file_path, error=str(e))
        
        # Also list files in workspace for context
        try:
            files = self.client.list_files(workspace)
            context_parts.append(f"=== Workspace files ===\n{', '.join(files[:50])}")
        except Exception:
            pass
            
        return "\n\n".join(context_parts) if context_parts else "No existing files found."

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def _execute_with_retry(self, prompt: str) -> str:
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
