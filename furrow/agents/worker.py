from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("task_started", task_id=self.task.id, description=self.task.description)

        # Read relevant files for context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = ""  # New file
            except Exception as e:
                file_contents[file_path] = f"Error reading: {e}"

        # Build prompt with file context
        files_context = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents.items()
        )

        prompt = f"""You are a worker agent in an autonomous coding system called Furrow.

Your job is to implement the assigned task completely by reading and writing files.

Task: {self.task.description}

Files to work on:
{', '.join(self.task.files) if self.task.files else 'Determine appropriate files'}

Current file contents:
{files_context if files_context else '(no existing files)'}

Instructions:
- Analyze the task and current files
- Return a JSON object with the files to create/update
- Format: {{"files": [{{"path": "file.py", "content": "..."}}]}}
- Only include files that need to be changed
- Write complete file contents, not diffs
- Make minimal, targeted changes
- Do not refactor unrelated code"""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse response and write files
        try:
            data = json.loads(response)
            files_to_write = data.get("files", [])
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat response as a single file if task.files has one item
            if len(self.task.files) == 1:
                files_to_write = [{"path": self.task.files[0], "content": response}]
            else:
                logger.error("worker_response_parse_failed", task_id=self.task.id, response=response[:200])
                return f"Failed to parse worker response: {response[:200]}..."

        # Write files
        for file_data in files_to_write:
            path = file_data.get("path")
            content = file_data.get("content", "")
            if path:
                await self.client.write_file(path, content)
                logger.debug("file_written", path=path, size=len(content))

        # Return summary
        changed = [f["path"] for f in files_to_write]
        logger.info("task_completed", task_id=self.task.id, files_changed=changed)
        return f"Modified {len(changed)} file(s): {', '.join(changed)}"