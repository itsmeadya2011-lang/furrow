"""Worker agent: execute one planned task against the real workspace.

The worker runs the agentic tool loop from :meth:`furrow.llm.LLMClient.complete_with_tools`
with the sandboxed :class:`~furrow.tools.WorkspaceTools` attached, so the task is
fulfilled by editing files rather than by describing an edit. The saved change on
disk is the deliverable.

``run()`` never raises for ordinary failures: a provider error, a tool error or a
prompt-formatting bug comes back as ``WorkerResult(ok=False, error=...)`` so that
one bad task cannot take the whole run down. ``asyncio.CancelledError`` is the one
exception and always propagates, because cancellation is a shutdown signal rather
than a task failure. Task state (``status``, ``attempts``, timings) is owned by the
orchestrator; the worker never mutates its :class:`~furrow.config.TaskModel`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import FIX_PROMPT, WORKER_PROMPT, WORKER_SYSTEM
from furrow.config import TaskModel
from furrow.config import settings as config_settings
from furrow.llm import LLMClient, ToolCall
from furrow.tools import WorkspaceTools

if TYPE_CHECKING:
    from furrow.config import Settings

_LOGGER = logging.getLogger(__name__)

#: Tools whose ``path`` argument counts as a change to the workspace.
WRITE_TOOLS = frozenset({"write_file", "edit_file"})

TRUNCATION_NOTE = (
    "NOTE: this response hit the model's output-token limit, so the work above is "
    "incomplete. Read back the files you were changing and finish them."
)

_NO_FILES = "(not specified - choose the files the task needs)"


@dataclass(slots=True)
class WorkerResult:
    """Outcome of one worker attempt."""

    ok: bool
    summary: str = ""
    tool_calls: int = 0
    files_changed: list[str] = field(default_factory=list)
    error: str | None = None
    iterations: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, suitable for event payloads."""
        return {
            "ok": self.ok,
            "summary": self.summary,
            "tool_calls": self.tool_calls,
            "files_changed": list(self.files_changed),
            "error": self.error,
            "iterations": self.iterations,
            "truncated": self.truncated,
        }


def _changed_files(tool_calls: list[ToolCall]) -> list[str]:
    """Sorted unique ``path`` arguments of the mutating tool calls."""
    paths: set[str] = set()
    for call in tool_calls:
        if call.name not in WRITE_TOOLS:
            continue
        arguments = getattr(call, "arguments", None)
        if not isinstance(arguments, dict):
            continue
        path = arguments.get("path")
        if isinstance(path, str) and path.strip():
            paths.add(path)
    return sorted(paths)


class WorkerAgent:
    """Runs a single :class:`~furrow.config.TaskModel` to completion."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient,
        *,
        settings: Settings | None = None,
        tools: WorkspaceTools | None = None,
        attempt: int = 1,
        feedback: str = "",
    ) -> None:
        self._task = task
        self._client = client
        self._settings = settings if settings is not None else config_settings
        self._tools = (
            tools if tools is not None else WorkspaceTools.from_settings(self._settings)
        )
        self._attempt = attempt
        self._feedback = feedback

    @property
    def task(self) -> TaskModel:
        """The task being executed (read-only view; never mutated)."""
        return self._task

    @property
    def tools(self) -> WorkspaceTools:
        """The sandboxed tools this worker may use."""
        return self._tools

    def _build_prompt(self) -> str:
        """Pick the repair prompt on a retry, the normal prompt otherwise."""
        if self._attempt > 1:
            return FIX_PROMPT.format(
                goal=self._task.description,
                failures=self._feedback,
                attempt=self._attempt,
            )
        files = ", ".join(self._task.files) if self._task.files else _NO_FILES
        return WORKER_PROMPT.format(
            description=self._task.description,
            files=files,
            attempt=self._attempt,
            feedback=self._feedback or "(none)",
        )

    async def run(self) -> WorkerResult:
        """Execute the task; never raises except on cancellation."""
        try:
            prompt = self._build_prompt()
            completion = await self._client.complete_with_tools(
                prompt,
                system=WORKER_SYSTEM,
                model=self._settings.model_for("worker"),
                tools=self._tools,
                max_iterations=self._settings.worker_max_iterations,
            )
        except asyncio.CancelledError:
            raise
        # Broad by design: one bad task must not kill the whole run.
        except Exception as exc:
            _LOGGER.warning("worker failed on task %s: %s", self._task.id, exc)
            # `str(exc)` is empty for message-less exceptions (TimeoutError(),
            # MemoryError(), ...); fall back to the type so the failure stays
            # diagnosable instead of reaching the reporter as "".
            return WorkerResult(ok=False, error=str(exc) or type(exc).__name__)

        summary = (completion.text or "").strip()
        if completion.truncated:
            summary = f"{summary}\n\n{TRUNCATION_NOTE}".strip()
        return WorkerResult(
            ok=True,
            summary=summary,
            tool_calls=len(completion.tool_calls),
            files_changed=_changed_files(completion.tool_calls),
            iterations=completion.iterations,
            truncated=completion.truncated,
        )
