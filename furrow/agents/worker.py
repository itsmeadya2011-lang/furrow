from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, TaskStatus
from furrow.config import get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.task.status = TaskStatus.IN_PROGRESS
        self.task.updated_at = __import__("datetime").datetime.utcnow().isoformat()

    async def run(self) -> str:
        # Read relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = "<file not found>"
            except OSError as e:
                file_contents[file_path] = f"<error reading: {e}>"

        files_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        ) if file_contents else "No existing files to reference."

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Existing file contents:
{files_context}

Return JSON only with this shape:
{{
  "changes": [
    {{"path": "file.py", "content": "full file content"}}
  ],
  "summary": "Brief description of changes made"
}}

If no files need to be changed, return empty changes array."""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            
            # Apply changes
            for change in changes:
                path = change.get("path")
                content = change.get("content", "")
                if path:
                    await self.client.write_file(path, content)
                    logger.info("file_written", path=path, size=len(content))
            
            logger.info("task_completed", task_id=self.task.id, summary=summary)
            return summary
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract useful info from response
            logger.warning("non_json_response", task_id=self.task.id, error=str(e))
            return f"Worker completed (non-JSON response): {response[:200]}"
