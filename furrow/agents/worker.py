from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_MAX_ATTEMPTS = 3
_RETRY_MESSAGE = (
    "\n\nYour previous attempt failed. Please retry, paying attention to the "
    "structured output format described above (files_changed, summary, issues)."
)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _build_prompt(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        dependencies = (
            ", ".join(self.task.dependencies) if self.task.dependencies else "none"
        )
        return (
            f"{WORKER_PROMPT}\n\n"
            f"Task ID: {self.task.id}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
            f"Dependencies: {dependencies}\n"
        )

    async def run(self) -> str:
        """Execute the assigned task and return the worker's raw response.

        The worker is instructed to return a structured result containing:
        - ``files_changed``: list of file paths modified
        - ``summary``: short description of what was done
        - ``issues``: list of any problems encountered (empty if none)

        The returned ``str`` is the raw LLM response, which should be parseable
        as that structured format (JSON or plain text per the prompt).

        The underlying LLM call is retried up to ``_MAX_ATTEMPTS`` times on
        failure, with a corrective note appended to the prompt between attempts.
        """
        prompt = self._build_prompt()
        last_error: BaseException | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return await self.client.complete(
                    prompt, model=self.client.settings.worker_model
                )
            except Exception as exc:
                last_error = exc
                if attempt >= _MAX_ATTEMPTS:
                    break
                prompt = f"{prompt}{_RETRY_MESSAGE}"
        assert last_error is not None
        raise last_error