from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.base import Agent
from furrow.agents.prompts import WORKER_PROMPT
from furrow.tools import DEFAULT_TOOLS
from furrow.tools.base import Tool

if TYPE_CHECKING:
    from furrow.config import TaskModel


class WorkerAgent(Agent):
    """Worker agent that implements a single task using tool-use.

    Unlike the original implementation that merely returned a text summary,
    this version gives the LLM access to file-reading, file-editing, and
    command-execution tools. The LLM iteratively reads relevant code,
    makes targeted edits, and runs tests until the task is complete.
    """

    system_prompt = WORKER_PROMPT
    tools = DEFAULT_TOOLS

    def __init__(self, task: "TaskModel", client=None, settings=None) -> None:
        super().__init__(client=client, settings=settings)
        self.task = task

    async def run(self) -> str:
        """Execute the task and return a summary."""
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            "Use your tools to read relevant files, make code changes, "
            "and verify with tests. Provide a concise summary when done."
        )
        model = self.client.settings.worker_model
        result = await self.run_loop(
            prompt,
            model=model,
            max_iterations=self.client.settings.agent_max_iterations,
        )
        return result.text
