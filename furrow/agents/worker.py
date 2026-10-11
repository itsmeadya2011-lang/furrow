from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


logger = get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description)
        
        # Build context with relevant files
        context_parts = [f"Task: {self.task.description}"]
        
        if self.task.files:
            context_parts.append("\nRelevant files:")
            for file_path in self.task.files:
                path = Path(file_path)
                if path.exists():
                    try:
                        content = await self.client.read_file(path)
                        context_parts.append(f"\n--- {file_path} ---\n{content}")
                    except OSError as e:
                        context_parts.append(f"\n--- {file_path} ---\n[Error reading file: {e}]")
                else:
                    context_parts.append(f"\n--- {file_path} ---\n[File does not exist, will be created]")

        prompt = f"{WORKER_PROMPT}\n\n" + "\n".join(context_parts) + "\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse response as file operations
        await self._apply_changes(response)
        
        logger.info("worker_completed", task_id=self.task.id)
        return response

    async def _apply_changes(self, response: str) -> None:
        """Parse response for file operations and apply them."""
        # Look for code blocks with file paths
        import re
        
        # Pattern: ```python\n# filepath: path/to/file.py\ncode...
        pattern = r"```(\w+)?\n# filepath: (.+?)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for lang, file_path, content in matches:
            file_path = file_path.strip()
            content = content.strip()
            logger.info("writing_file", task_id=self.task.id, file=file_path)
            try:
                await self.client.write_file(file_path, content)
            except OSError as e:
                logger.error("file_write_failed", task_id=self.task.id, file=file_path, error=str(e))
