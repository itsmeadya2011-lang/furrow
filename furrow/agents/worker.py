from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        # Resolve settings once so the class does not reach through
        # self.client.settings for every field on every call.
        self.settings = self.client.settings

    async def run(self, feedback: str = "") -> str:
        """Return the worker's textual report for the task.

        The worker currently has no tool access and cannot edit files, so its
        output is a description of what needs to change, not a claim that the
        change was made.  An empty or whitespace-only result is treated as a
        failure: the orchestrator records it as ``completed`` today, which
        silently turns a refusal into a pass.

        Raises:
            ValueError: if the LLM returns an empty or whitespace-only result.
        """
        prompt = _build_prompt(self.task, feedback, self.settings.workspace)
        result = await self.client.complete(prompt, model=self.settings.worker_model)
        if not result.strip():
            raise ValueError("Worker returned an empty result")
        return result


def _build_prompt(task: TaskModel, feedback: str, workspace) -> str:
    sections = [
        WORKER_PROMPT,
        f"Task: {task.description}",
        f"Files to touch: {', '.join(task.files) if task.files else 'any'}",
        f"Absolute workspace path: {workspace}",
    ]
    if feedback:
        sections.append(
            "Failure output from the previous cycle (describe how you would address it):\n"
            + feedback
        )
    return "\n\n".join(sections)
