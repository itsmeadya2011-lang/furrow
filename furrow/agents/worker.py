from __future__ import annotations

from typing import TYPE_CHECKING, Final

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

#: Cap on the repo file list embedded in the prompt so it cannot explode.
MAX_REPO_FILES: Final[int] = 200


class WorkerAgent:
    """Implements a single planned task and returns the worker's report."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        """Return the worker's summary of the work done for ``self.task``."""
        prompt = "\n\n".join(
            part for part in (WORKER_PROMPT, self._task_block(), self._repo_block()) if part
        )
        return await self.client.complete(
            prompt, system=WORKER_SYSTEM_PROMPT, model=self.client.settings.worker_model
        )

    def _task_block(self) -> str:
        """Describe the assigned task, its files and its dependencies."""
        task = self.task
        files = ", ".join(task.files) if task.files else "any"
        dependencies = ", ".join(task.dependencies) if task.dependencies else "nothing"
        return (
            f"Task id: {task.id}\n"
            f"Task: {task.description}\n"
            f"Files to touch: {files}\n"
            f"Depends on: {dependencies}"
        )

    def _repo_block(self) -> str:
        """List the current repository files so the worker knows the codebase layout."""
        files = self.client.list_files()
        listed = "\n".join(files[:MAX_REPO_FILES])
        if len(files) > MAX_REPO_FILES:
            listed = f"{listed}\n... (list truncated at {MAX_REPO_FILES} files)"
        return f"Repository files:\n{listed}"