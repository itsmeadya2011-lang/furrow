from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    """Worker agent that executes a single task using LLM tool calling.

    The worker is given access to file read/write, directory listing, and
    shell execution tools.  It runs an LLM agent loop until the task is
    complete or the maximum iteration count is reached.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        """Execute the task and return a summary of what was done."""
        files_str = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task ID: {self.task.id}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
            f"Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n"
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        system = (
            "You are a worker agent in an autonomous coding system called Furrow. "
            "Your job is to implement the assigned task completely and concisely. "
            "You have access to tools for reading/writing files, listing directories, "
            "and executing shell commands. Use them to complete the task. "
            "Make minimal, targeted changes. Do not refactor unrelated code. "
            "After you finish, return a concise summary of what you changed."
        )

        logger.info("Worker started", task_id=self.task.id, description=self.task.description)
        result = await self.client.complete_with_tools(
            messages=messages,
            system=system,
            model=self.client.settings.worker_model,
        )
        logger.info("Worker finished", task_id=self.task.id)
        return result
