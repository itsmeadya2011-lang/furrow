from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = str(self.client.settings.workspace)
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Workspace path: {workspace}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        try:
            return await asyncio.wait_for(
                self.client.complete(prompt, model=self.client.settings.worker_model),
                timeout=120,
            )
        except asyncio.TimeoutError:
            return (
                f"Error: Worker timed out after 120 seconds while calling LLM "
                f"for task: {self.task.description}"
            )
        except Exception as e:
            return (
                f"Error: LLM call failed for task '{self.task.description}': "
                f"{type(e).__name__}: {e}"
            )
