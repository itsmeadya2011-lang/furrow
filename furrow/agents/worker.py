"""The worker agent: the component that actually changes the repository.

A worker owns one :class:`~furrow.config.TaskModel` and drives a tool loop:
:class:`~furrow.agents.tools.ToolRegistry` exposes the sandbox, the model asks for
tools, this module executes them and feeds the output back until the model stops.

Constructor shape is a contract with the orchestrator:
``WorkerAgent(task=task, client=client, settings=settings, bus=bus)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

from furrow.agents.prompts import WORKER_SYSTEM_PROMPT
from furrow.agents.tools import ToolContext, ToolRegistry
from furrow.config import Settings, TaskModel, TaskStatus, get_settings
from furrow.events import EventBus, EventType
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.llm import ToolCall, ToolResult

MAX_ORIENTATION_CHARS = 3_000
MAX_TRANSCRIPT_CHARS = 12_000
MAX_BLOCK_CHARS = 3_000
MAX_SUMMARY_CHARS = 4_000


def _clip(text: str, limit: int) -> str:
    """Cap `text` at `limit` chars with a visible marker, so nothing is silently lost."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters truncated] ..."


def _trim(blocks: Sequence[str], budget: int) -> list[str]:
    """Drop the oldest tool outputs until the transcript fits inside `budget`."""
    kept = list(blocks)
    while len(kept) > 1 and sum(len(block) for block in kept) > budget:
        kept.pop(0)
    return kept


class WorkerAgent:
    """Runs one task to completion using the sandboxed tool registry."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.task = task
        self.settings: Settings = settings or getattr(client, "settings", None) or get_settings()
        self.client: LLMClient = client if client is not None else LLMClient(
            settings=self.settings
        )
        self.bus = bus

    async def run(self) -> str:
        """Execute the assigned task using tools. Returns a concise summary.

        Mutates ``self.task``: ``status`` becomes IN_PROGRESS while the loop runs and then
        COMPLETED or FAILED, the summary is stored in ``self.task.result``, and
        ``self.task.attempts`` is incremented once per call.

        Failure policy: nothing propagates except ``asyncio.CancelledError``. A crash in
        the loop - including an unusable workspace or a client without tool support - is
        returned as a descriptive failure string with ``status = FAILED``, so an
        orchestrator's ``asyncio.gather`` still sees a result for every worker. Cancellation
        is re-raised because swallowing it would break shutdown.

        Exhausting ``settings.max_tool_iterations`` is not a crash: the summary says so
        explicitly and the task is left COMPLETED, since partial file changes are already on
        disk and the orchestrator should surface the summary rather than re-run the task.
        """
        self.task.status = TaskStatus.IN_PROGRESS
        self.task.attempts += 1
        self._emit(
            EventType.TASK_START,
            f"task {self.task.id}: {self.task.description}",
            task_id=self.task.id,
            files=list(self.task.files),
            attempt=self.task.attempts,
        )
        registry: ToolRegistry | None = None
        try:
            registry = ToolRegistry(
                ToolContext(
                    workspace=self.settings.resolved_workspace(),
                    settings=self.settings,
                    bus=self.bus,
                )
            )
            summary = await self._drive(registry)
        except asyncio.CancelledError:
            self.task.status = TaskStatus.FAILED
            self.task.result = "Task cancelled before it finished."
            self._emit(EventType.ERROR, "task cancelled", task_id=self.task.id)
            raise
        except Exception as exc:
            self.task.status = TaskStatus.FAILED
            self.task.result = f"Worker failed before finishing: {exc}"
            self._emit(EventType.ERROR, str(exc), task_id=self.task.id)
            return self.task.result
        else:
            self.task.status = TaskStatus.COMPLETED
            self.task.result = summary
            return summary
        finally:
            self._emit(
                EventType.TASK_END,
                f"task {self.task.id}: {self.task.status.value}",
                task_id=self.task.id,
                status=self.task.status.value,
                changed=list(registry.changed_files) if registry is not None else [],
            )

    async def _drive(self, registry: ToolRegistry) -> str:
        """Round-trip with the model, executing tools until it stops asking for them.

        ``complete_with_tools`` is called with ``max_iterations=1`` so this module owns the
        loop and the transcript: each round is one model call whose prompt is rebuilt from
        the task framing plus the tool outputs collected so far.
        """
        complete = self._complete_with_tools()
        log: list[tuple[str, str]] = []
        transcript: list[str] = []

        async def on_tool_call(call: ToolCall) -> None:
            """Execute one requested tool and log its output for the next round.

            ``invoke`` never raises, so the model always gets feedback it can act on.
            """
            output = await registry.invoke(call.name, call.arguments)
            log.append((call.name, output))

        orientation = _clip(
            await registry.invoke("list_files", {"path": ".", "pattern": "*"}),
            MAX_ORIENTATION_CHARS,
        )
        rounds = max(1, int(self.settings.max_tool_iterations or 1))
        text, stop_reason = "", "end_turn"
        for _ in range(rounds):
            seen = len(log)
            result = await complete(
                self._compose(registry, orientation, transcript),
                registry.specs(),
                system=WORKER_SYSTEM_PROMPT,
                model=self.settings.worker_model,
                max_iterations=1,
                on_tool_call=on_tool_call,
            )
            text, stop_reason = result.text, result.stop_reason
            if stop_reason == "end_turn" or len(log) == seen:
                break  # the model stopped asking for tools: it is done
            for name, output in log[seen:]:
                transcript.append(f"[{name} ->]\n{_clip(output, MAX_BLOCK_CHARS)}")
            transcript = _trim(transcript, MAX_TRANSCRIPT_CHARS)
        return self._summarize(registry, text, stop_reason, transcript)

    def _complete_with_tools(self) -> Callable[..., Awaitable[ToolResult]]:
        """Return the client's tool-calling entry point.

        Resolved dynamically so a client without tool support fails loudly here, with a
        useful message, instead of raising ``AttributeError`` from inside the loop.
        """
        method = getattr(self.client, "complete_with_tools", None)
        if method is None:
            raise RuntimeError(
                f"{type(self.client).__name__} has no complete_with_tools; WorkerAgent "
                "requires a tool-calling LLM client."
            )
        return cast("Callable[..., Awaitable[ToolResult]]", method)

    def _compose(
        self, registry: ToolRegistry, orientation: str, transcript: Sequence[str]
    ) -> str:
        """Build the user message: the task, its target files and the workspace layout.

        The role and tool rules live in ``WORKER_SYSTEM_PROMPT``, which is passed as the
        system message on every round, so they are not repeated here.
        """
        targets = ", ".join(self.task.files) if self.task.files else "none listed"
        head = (
            f"Task {self.task.id}: {self.task.description}\n"
            f"Target files: {targets}\n"
            f"Workspace root: {registry.workspace} - every tool path is relative to it.\n\n"
            f"Workspace contents (workspace-relative paths):\n{orientation}\n\n"
            "Do the work with the tools now. Reply with a plain-text summary, and no tool "
            "call, once the task is done."
        )
        if not transcript:
            return head
        return f"{head}\n\nTool results so far:\n" + "\n\n".join(transcript)

    def _summarize(
        self,
        registry: ToolRegistry,
        text: str,
        stop_reason: str,
        transcript: Sequence[str],
    ) -> str:
        """Build the run summary. Always names the files that actually changed."""
        notes = [_clip(text.strip() or "(the model returned no summary)", MAX_SUMMARY_CHARS)]
        if stop_reason == "max_iterations":
            notes.insert(
                0,
                "INCOMPLETE: the tool iteration limit was reached, so this task is NOT "
                "finished. Treat the changes below as work in progress.",
            )
        elif stop_reason not in {"end_turn", ""}:
            notes.insert(
                0, f"The model turn ended with stop_reason={stop_reason!r}; verify the work."
            )
        changed = registry.changed_files
        notes.append(f"Files changed: {', '.join(changed) if changed else 'none'}")
        if transcript:
            notes.append(f"Tool results collected: {len(transcript)}")
        return "\n".join(notes)

    def _emit(self, event: EventType, message: str, **data: Any) -> None:
        """Emit to the bus when one was supplied. Workers are usable without a bus."""
        if self.bus is not None:
            self.bus.emit(event, message, **data)
