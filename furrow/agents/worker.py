from __future__ import annotations

import json
import logging

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        codebase_context: str = "",
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())
        self.codebase_context = codebase_context

    async def run(self) -> str:
        # Read relevant files if specified
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:  # noqa: BLE001
                file_contents[file_path] = f"[Error reading {file_path}: {e}]"

        files_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        ) if file_contents else "No specific files to reference."

        prompt = f"""{WORKER_PROMPT}

Codebase Context:
{self.codebase_context}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Relevant file contents:
{files_context}

Return a JSON object with:
- "changes": list of {{"file": "...", "content": "..."}} for files to create/modify
- "summary": brief description of what was done
- "complete": boolean indicating if task is fully complete"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            changes = data.get("changes", [])
            summary = data.get("summary", "Task completed")
            _complete = data.get("complete", True)

            # Apply changes
            for change in changes:
                file_path = change.get("file")
                content = change.get("content", "")
                if file_path:
                    await self.client.write_file(file_path, content)

            return summary
        except (json.JSONDecodeError, ValueError):
            # Fallback: return the raw response as summary
            return response
