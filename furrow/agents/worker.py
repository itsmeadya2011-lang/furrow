from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import DEFAULT_TOOLS, LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Executes a single task using an LLM with tool-use capability.

    The worker sends the task description to the LLM with access to file
    read/write, command execution, glob, and grep tools.  It iterates
    until the LLM completes the task or the iteration limit is reached.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        tools: list[dict] | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.tools = tools
        self.max_iterations = max_iterations

    async def run(self) -> str:
        # Augment the worker prompt with task context
        files_str = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task ID: {self.task.id}\n"
            f"Task Description: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"Implement this task using the available tools. "
            f"Read files as needed, make changes, and verify with tests. "
            f"Return a concise summary of what you changed and any issues."
        )

        system = (
            "You are a Furrow worker agent. Your job is to implement the assigned "
            "task completely and concisely. You have access to tools for reading "
            "and writing files, running shell commands, searching, and listing. "
            "Work only on the assigned task. Do not refactor unrelated code. "
            "Return a concise summary of what you changed and any issues."
        )

        return await self.client.complete_with_tools(
            prompt=prompt,
            system=system,
            model=self.client.settings.worker_model,
            tools=self.tools or DEFAULT_TOOLS,
            max_iterations=self.max_iterations,
        )
