from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger("furrow.worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Read relevant files first to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                file_contents[file_path] = f"[Error reading file: {e}]"

        # Build prompt with file context
        context_parts = []
        for path, content in file_contents.items():
            context_parts.append(f"--- {path} ---\n{content}")
        
        file_context = "\n\n".join(context_parts) if context_parts else "No existing files to reference."
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nExisting file contents:\n{file_context}\n\nReturn a JSON object with:\n- \"changes\": array of {{ \"file\": \"path\", \"content\": \"new content\" }}\n- \"summary\": brief description of what was done\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content")
                if file_path and content is not None:
                    await self.client.write_file(file_path, content)
                    logger.info(f"Wrote to {file_path}")
            
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat response as summary if not JSON
            logger.warning(f"Worker returned non-JSON response: {e}")
            return response
