"""Worker agent for the Furrow autonomous coding system.

This module defines :class:`WorkerAgent`, which executes a single task by
building a prompt from the task description and delegating to an LLM client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """An agent that executes a single task using an LLM.

    The worker agent is given a task description and a list of files to touch.
    It builds a prompt from the task details and delegates the actual work to
    an :class:`~furrow.llm.LLMClient`.
    """
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
