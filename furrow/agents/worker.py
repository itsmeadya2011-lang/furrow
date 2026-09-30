from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger("furrow.worker")


class WorkerAgent:
    """Executes a single planned task, using file tools to make real changes.

    The worker asks the LLM to implement the task.  The LLM may call
    ``read_file`` / ``write_file`` / ``list_files`` tools during its turn and
    the client executes those against the real filesystem (constrained to the
    workspace).  The final text response — a concise summary — is returned.
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
        files = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n\n"
            f"If dependencies are listed, complete them first.\n"
            f"Use read_file/write_file/list_files tools to make the changes, "
            f"then finish with a concise summary."
        )
        log.debug("worker.run", task_id=self.task.id, files=files)
        try:
            summary = await self.client.complete_with_tools(
                prompt,
                system="You are a meticulous coding agent. Make minimal, targeted, "
                "correct changes only within the assigned task. Always return a final "
                "text summary of what you changed.",
                model=self.client.settings.worker_model,
            )
            return summary
        except Exception as exc:
            log.error("worker.run_failed", task_id=self.task.id, error=str(exc))
            raise
