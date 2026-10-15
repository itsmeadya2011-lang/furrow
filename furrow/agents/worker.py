from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileOperation, TaskModel
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
        # Read existing files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""  # New file

        # Build context from existing files
        context = ""
        if file_contents:
            context = "\n\nExisting files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            operations = data.get("operations", [])
            summary = data.get("summary", "Task completed")
            
            # Validate and execute file operations
            for op_data in operations:
                op = FileOperation(**op_data)
                if op.operation == "write":
                    await self.client.write_file(op.path, op.content)
                elif op.operation == "append":
                    existing = await self.client.read_file(op.path)
                    await self.client.write_file(op.path, existing + op.content)
                elif op.operation == "delete":
                    import os
                    os.remove(op.path)
                else:
                    logger.warning("unknown_operation", operation=op.operation)
            
            logger.info("worker_completed", task_id=self.task.id, operations=len(operations))
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("worker_parse_failed", task_id=self.task.id, error=str(e), response=response[:500])
            # Fallback: return the raw response as summary
            return f"Completed (raw): {response[:200]}"
