from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description)
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Return JSON only (no markdown, no explanation) with this shape:
{{
  "summary": "Brief summary of changes",
  "files": [
    {{
      "path": "relative/path/to/file",
      "content": "full file content here"
    }}
  ]
}}

If you cannot complete the task, return an error in the summary and an empty files array.
"""
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(response)
            summary = data.get("summary", response)
            files = data.get("files", [])
            written = []
            for file_info in files:
                path = file_info.get("path")
                content = file_info.get("content", "")
                if path:
                    await self.client.write_file(path, content)
                    written.append(path)
            if written:
                summary += f"\nWritten files: {', '.join(written)}"
            logger.info("worker_complete", task_id=self.task.id, written_files=len(written))
            return summary
        except (json.JSONDecodeError, ValueError):
            logger.warning("worker_json_parse_failed", task_id=self.task.id, response=response)
            return response
