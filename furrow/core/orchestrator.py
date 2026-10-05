"""The Furrow development loop: plan, execute in parallel, test, repeat.

One cycle is:

1. Ask :class:`~furrow.agents.planner.PlannerAgent` for a
   :class:`~furrow.config.Plan` built from the **original** goal plus the
   current ``focus`` (a short instruction describing what this cycle should
   concentrate on, which becomes a fix instruction after a failing test run).
2. Execute that plan in dependency-ordered waves.  A task only starts once
   every task it depends on has completed; tasks sharing a wave run
   concurrently under an :class:`asyncio.Semaphore` sized by
   ``Settings.max_parallel_tasks``, which also caps how many tasks a single
   plan may contain.
3. Run :class:`~furrow.agents.tester.TesterAgent` over everything accumulated
   so far and fold the verdict back into ``focus``.

The loop is observable.  Every transition is published through the synchronous
``on_event`` callback as ``callback(event_name, payload_dict)``:

==================  ====================================================
event               payload keys
==================  ====================================================
``cycle_start``     ``cycle``, ``goal``
``plan``            ``cycle``, ``plan``, ``rationale``
``task_start``      ``cycle``, ``task_id``, ``description``
``task_complete``   ``cycle``, ``task_id``, ``result``
``task_failed``     ``cycle``, ``task_id``, ``error``
``task_blocked``    ``cycle``, ``task_id``, ``reason``
``tests``           ``cycle``, ``passed``, ``summary``, ``failures``
``cycle_end``       ``cycle``, ``done``, ``reason``
``error``           ``cycle``, ``message``
``halt``            ``cycles``, ``reason``, ``done``
==================  ====================================================

``halt`` is emitted exactly once, on every exit path: goal complete, the
``max_cycles`` cap, a planner error, cancellation, and unexpected failures.
Every payload is JSON serialisable, so a consumer may forward it verbatim over
a WebSocket.  The rich console output is rendered from those same events, which
keeps the CLI view and the web view from drifting apart; the callback itself is
wrapped so a broken consumer can never take the loop down.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import structlog
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

__all__ = ["EVENT_NAMES", "EventCallback", "Orchestrator", "plan_waves"]

#: Signature of the synchronous observer the orchestrator publishes to.
EventCallback = Callable[[str, dict[str, Any]], None]

#: Every event name the orchestrator may emit, for consumer-side validation.
EVENT_NAMES: frozenset[str] = frozenset(
    {
        "cycle_start",
        "plan",
        "task_start",
        "task_complete",
        "task_failed",
        "task_blocked",
        "tests",
        "cycle_end",
        "error",
        "halt",
    }
)

log = structlog.get_logger(__name__)

console = Console()

_PENDING = "pending"
_IN_PROGRESS = "in_progress"
_COMPLETED = "completed"
_FAILED = "failed"
_BLOCKED = "blocked"

#: Default when ``Settings.max_parallel_tasks`` is absent or unusable.
_DEFAULT_MAX_PARALLEL_TASKS = 5
#: Default when ``Settings.max_cycles`` is absent or unusable.
_DEFAULT_MAX_CYCLES = 0
#: Semaphore width used when the caller disables the parallel cap.
_UNLIMITED_TASKS = 1_000_000
#: Test failures folded into ``focus``; the rest are only counted.
_MAX_FOCUS_FAILURES = 20
#: Characters kept from an exception message placed in an event payload.
_MAX_ERROR_CHARS = 500
#: Failures echoed to the console per test run.
_MAX_PRINTED_FAILURES = 10

_configured = False


def _configure_logging(level: str) -> None:
    """Configure structlog from ``level`` exactly once per process.

    The flag is set before the call so that repeated orchestration runs, or a
    module imported twice under different names, never reconfigure global
    logging; the first caller wins.

    Args:
        level: A stdlib level name such as ``"DEBUG"`` or ``"info"``.
    """
    global _configured
    if _configured:
        return
    _configured = True
    numeric = getattr(logging, str(level).strip().upper(), logging.INFO)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    factory = getattr(structlog, "PrintLoggerFactory", None)
    wrapper = getattr(structlog, "make_filtering_bound_logger", None)
    try:
        structlog.configure(
            wrapper_class=wrapper(numeric) if callable(wrapper) else structlog.BoundLogger,
            logger_factory=factory() if callable(factory) else structlog.PrintLogger,
            cache_logger_on_first_use=True,
        )
    except Exception as exc:  # pragma: no cover - logging must never break a run
        log.debug("logging_configuration_failed", error=str(exc)[:_MAX_ERROR_CHARS])


def plan_waves(plan: Plan) -> list[list[TaskModel]]:
    """Group plan tasks into dependency-ordered execution levels.

    A task lands in level *n* only when every id in its ``dependencies`` that
    also appears in the plan was placed in an earlier level, so all tasks of a
    level are safe to run concurrently and levels are strictly ordered.
    Dependencies on ids outside the plan, and self-dependencies, are treated as
    satisfied.  A genuine dependency cycle cannot be reduced; the remaining
    tasks are then appended as one final level so the caller still runs
    everything, and a warning is logged.  The function always terminates.

    Args:
        plan: The plan to order.  Any object exposing ``tasks`` whose items
            carry ``id`` and ``dependencies`` attributes will do.

    Returns:
        A list of levels, each a list of tasks, in execution order.  An empty
        plan yields an empty list.
    """
    tasks = list(plan.tasks)
    if not tasks:
        return []

    position: dict[str, int] = {}
    for index, task in enumerate(tasks):
        position.setdefault(str(task.id), index)

    depends: dict[int, list[int]] = {}
    for index, task in enumerate(tasks):
        wanted: list[int] = []
        for dependency in _dependency_ids(task):
            target = position.get(dependency)
            if target is None or target == index or target in wanted:
                continue
            wanted.append(target)
        depends[index] = wanted

    waves: list[list[TaskModel]] = []
    settled: set[int] = set()
    remaining = set(range(len(tasks)))
    while remaining:
        ready = sorted(index for index in remaining if all(d in settled for d in depends[index]))
        if not ready:
            log.warning("plan_dependency_cycle", pending=len(remaining), depth=len(waves))
            waves.append([tasks[index] for index in sorted(remaining)])
            break
        waves.append([tasks[index] for index in ready])
        settled.update(ready)
        remaining.difference_update(ready)
    return waves


class Orchestrator:
    """Drive the plan/execute/test cycle until the goal is met.

    The original ``goal`` is never rewritten.  What a cycle should concentrate
    on lives in :attr:`focus`, and everything seen so far accumulates in
    :attr:`tasks`, so no cycle re-plans against a truncated goal or forgets what
    already succeeded.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Bind the goal to a client, settings and an event observer.

        Args:
            goal: The objective; kept verbatim for the whole run.
            client: Shared :class:`~furrow.llm.LLMClient`.  One is created here
                when omitted, and only then is it closed at the end of a run.
            on_event: Synchronous ``callback(name, payload)`` observer.  It is
                invoked inside a ``try``/``except``, so an exception in the
                consumer is logged and swallowed.
            settings: Explicit settings.  Defaults to the client's own
                settings, then to the module singleton.
        """
        self.goal = str(goal)
        self._owns_client = client is None
        self.client = client if client is not None else LLMClient(settings=settings)
        self.settings = _resolve_settings(self.client, settings)
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client, settings=self.settings)

        self.focus = self.goal
        self.tasks: list[TaskModel] = []
        self.test_results: list[TestResult] = []
        self.cycles = 0
        self.done = False
        self.halt_reason = ""

        self._running = False
        self._halted = False
        self._empty_plans = 0
        self._plan_was_empty = False
        self._pending: set[asyncio.Task[Any]] = set()

        _configure_logging(getattr(self.settings, "log_level", "INFO"))
        log.info(
            "orchestrator_created",
            goal_chars=len(self.goal),
            max_parallel_tasks=self._parallel_cap(),
            max_cycles=self._cycle_cap(),
        )

    @property
    def status(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot for the CLI and web layers.

        The shape is stable: ``goal``, ``focus``, ``cycles``, ``done``,
        ``halt_reason``, ``running``, a ``tasks`` list of ``{id, description,
        status, files, result}`` objects, and ``last_test`` which is ``None``
        before the first run or a ``{passed, summary, failures}`` object.
        """
        last = self.test_results[-1] if self.test_results else None
        return {
            "goal": self.goal,
            "focus": self.focus,
            "cycles": int(self.cycles),
            "done": bool(self.done),
            "halt_reason": self.halt_reason,
            "running": bool(self._running),
            "tasks": [
                {
                    "id": str(task.id),
                    "description": str(task.description),
                    "status": _status(task),
                    "files": [str(item) for item in getattr(task, "files", None) or []],
                    "result": _optional_text(getattr(task, "result", None)),
                }
                for task in self.tasks
            ],
            "last_test": None
            if last is None
            else {
                "passed": bool(last.passed),
                "summary": str(last.summary),
                "failures": [str(item) for item in last.failures],
            },
        }

    async def run(self) -> None:
        """Run the development loop until the goal is met or a stop is required.

        Emits exactly one ``halt`` event before returning, whatever the exit
        path.  On cancellation the ``halt`` reason is ``"cancelled"``, every
        in-flight worker task is awaited, and :class:`asyncio.CancelledError` is
        re-raised so the caller keeps control.

        Returns:
            ``None``.  Inspect :attr:`status` for the outcome.
        """
        self._running = True
        self.done = False
        self.halt_reason = ""
        self._halted = False
        semaphore = asyncio.Semaphore(self._parallel_cap())
        self._print_banner()
        try:
            while True:
                cap = self._cycle_cap()
                if cap > 0 and self.cycles >= cap:
                    self._halt("max_cycles", done=False)
                    return

                self.cycles += 1
                cycle = self.cycles
                self._emit("cycle_start", cycle=cycle, goal=self.goal)
                try:
                    stop = await self._run_cycle(cycle, semaphore)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    reason = "unexpected_error"
                    self._fail(cycle, exc)
                    self._emit_cycle_end(cycle, False, reason)
                    self._halt(reason, done=False)
                    return
                if stop:
                    self._halt(stop, done=False)
                    return
                if self._is_done():
                    self._halt(self._done_reason(), done=True)
                    return
        except asyncio.CancelledError:
            self._halt("cancelled", done=False)
            raise
        finally:
            try:
                await self._drain()
            finally:
                self._running = False
                await self._close_client()

    async def _run_cycle(self, cycle: int, semaphore: asyncio.Semaphore) -> str:
        """Run one plan/execute/test cycle.

        Args:
            cycle: The 1-based cycle number, for event payloads.
            semaphore: Bounds how many workers may be in flight at once.

        Returns:
            An empty string to keep looping, or a halt reason when the cycle
            cannot continue at all (currently only ``"planner_error"``).
        """
        try:
            plan = await self.planner.plan(self._planner_prompt())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(cycle, exc)
            self._emit_cycle_end(cycle, False, "planner_error")
            return "planner_error"

        self._plan_was_empty = not plan.tasks
        self._empty_plans = self._empty_plans + 1 if self._plan_was_empty else 0
        capped = self._cap_plan(plan, cycle)
        self._emit(
            "plan",
            cycle=cycle,
            plan=capped.model_dump(mode="json"),
            rationale=str(capped.rationale),
        )
        if self._plan_was_empty:
            done = self._is_done()
            self._emit_cycle_end(cycle, done, self._done_reason() if done else "continue")
            return ""

        self._merge_plan(capped)
        await self._execute(capped, cycle, semaphore)
        await self._test(cycle)
        done = self._is_done()
        self._emit_cycle_end(cycle, done, self._done_reason() if done else "continue")
        return ""

    async def _execute(self, plan: Plan, cycle: int, semaphore: asyncio.Semaphore) -> None:
        """Run every wave of ``plan`` in order, tasks within a wave concurrently."""
        for level in plan_waves(plan):
            await self._run_level(plan, level, cycle, semaphore)

    async def _run_level(
        self,
        plan: Plan,
        level: list[TaskModel],
        cycle: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Run one wave, blocking tasks whose dependency did not complete.

        Args:
            plan: The plan the wave came from, used to resolve dependencies.
            level: The tasks of a single wave.
            cycle: The 1-based cycle number, for event payloads.
            semaphore: Bounds how many workers may be in flight at once.
        """
        scheduled: list[TaskModel] = []
        handles: list[asyncio.Task[Any]] = []
        for task in level:
            dependency = self._failed_dependency(plan, task)
            if dependency is not None:
                task.status = _BLOCKED
                task.result = f"Blocked: dependency {dependency} did not complete"
                self._emit("task_blocked", cycle=cycle, task_id=task.id, reason=task.result)
                continue
            task.status = _IN_PROGRESS
            self._emit(
                "task_start",
                cycle=cycle,
                task_id=task.id,
                description=task.description,
            )
            handle = asyncio.create_task(
                self._run_task(task, semaphore),
                name=f"furrow-task-{cycle}-{task.id}",
            )
            handles.append(handle)
            self._pending.add(handle)
            scheduled.append(task)

        if not handles:
            return
        results = await asyncio.gather(*handles, return_exceptions=True)
        self._pending.difference_update(handles)
        for task, result in zip(scheduled, results):
            if isinstance(result, BaseException):
                task.status = _FAILED
                task.result = _error_text(result)
                self._emit("task_failed", cycle=cycle, task_id=task.id, error=task.result)
            else:
                task.status = _COMPLETED
                task.result = str(result)
                self._emit(
                    "task_complete",
                    cycle=cycle,
                    task_id=task.id,
                    result=task.result,
                )

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        """Run one worker behind the semaphore and return its summary."""
        async with semaphore:
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            return await worker.run()

    async def _test(self, cycle: int) -> TestResult:
        """Run the tester, synthesise a failure on error, and update ``focus``.

        Args:
            cycle: The 1-based cycle number, for the ``tests`` payload.

        Returns:
            The test verdict; a broken tester degrades to ``passed=False``.
        """
        tester = TesterAgent(client=self.client, settings=self.settings)
        try:
            result = await tester.run(self.goal, self.tasks)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            log.warning("tester_failed", cycle=cycle, error=message[:_MAX_ERROR_CHARS])
            result = TestResult(
                passed=False,
                summary=f"Tester agent failed: {message}"[:_MAX_ERROR_CHARS],
                failures=[message],
            )
        failures = [str(item) for item in result.failures]
        self.test_results.append(result)
        self._emit(
            "tests",
            cycle=cycle,
            passed=bool(result.passed),
            summary=str(result.summary),
            failures=failures,
        )
        if not result.passed:
            self.focus = self._fix_instruction(failures)
        return result

    async def _drain(self) -> None:
        """Await every in-flight worker so that no task is left un-awaited."""
        outstanding = [handle for handle in self._pending if not handle.done()]
        if outstanding:
            await asyncio.gather(*outstanding, return_exceptions=True)
        self._pending.clear()

    async def _close_client(self) -> None:
        """Close the LLM client, but only when this orchestrator created it."""
        if not self._owns_client:
            return
        closer = getattr(self.client, "aclose", None)
        if not callable(closer):
            return
        try:
            await closer()
        except Exception as exc:  # pragma: no cover - teardown must not raise
            log.warning("client_close_failed", error=str(exc)[:_MAX_ERROR_CHARS])

    def _is_done(self) -> bool:
        """Report whether the goal is met and the loop may stop.

        Returns:
            ``True`` when the planner reported no work and nothing is
            outstanding, or when every accumulated task completed and the most
            recent test run passed.  ``False`` whenever any task is failed,
            blocked or unfinished, and ``False`` when tests never ran.
        """
        outstanding = any(not task.succeeded for task in self.tasks)
        if self._plan_was_empty:
            if not outstanding:
                return True
            return self._empty_plans >= 2
        if outstanding:
            return False
        if not self.tasks or not self.test_results:
            return False
        if not all(task.succeeded for task in self.tasks):
            return False
        return bool(self.test_results[-1].passed)

    def _get_tasks(self) -> list[TaskModel]:
        """Return every task seen across all cycles (the live accumulator)."""
        return self.tasks

    def _done_reason(self) -> str:
        """Name the condition that ended the loop successfully."""
        return "no_tasks" if self._plan_was_empty else "goal_complete"

    def _planner_prompt(self) -> str:
        """Combine the original goal with the current focus for the planner."""
        if not self.focus or self.focus == self.goal:
            return self.goal
        return f"{self.goal}\n\nCurrent focus for this cycle:\n{self.focus}"

    def _cap_plan(self, plan: Plan, cycle: int) -> Plan:
        """Truncate a plan to ``max_parallel_tasks`` tasks, logging the cut."""
        cap = self._parallel_cap()
        if not cap or len(plan.tasks) <= cap:
            return plan
        log.warning("plan_truncated", cycle=cycle, planned=len(plan.tasks), cap=cap)
        return Plan(tasks=list(plan.tasks[:cap]), rationale=plan.rationale)

    def _merge_plan(self, plan: Plan) -> None:
        """Accumulate a plan's tasks, de-duplicating by id.

        A re-planned id adopts the fresh model, but a status of ``completed`` is
        carried forward when that id previously succeeded, so earlier work is
        never forgotten.  Tasks the current plan does not mention keep their
        place at the end of the accumulator.  Tasks dropped by the parallel cap
        are deliberately not accumulated: they were never scheduled.
        """
        previous: dict[str, TaskModel] = {}
        for task in self.tasks:
            previous.setdefault(str(task.id), task)
        merged: list[TaskModel] = []
        for fresh in plan.tasks:
            done = previous.pop(str(fresh.id), None)
            if done is not None and _status(done) == _COMPLETED:
                fresh.status = _COMPLETED
                fresh.result = done.result
            merged.append(fresh)
        merged.extend(previous.values())
        self.tasks = merged

    def _fix_instruction(self, failures: list[str]) -> str:
        """Build the next cycle's focus from a failing test run."""
        shown = failures[:_MAX_FOCUS_FAILURES]
        lines = [f"- {item}" for item in shown] or ["- the suite reported no detail"]
        if len(failures) > len(shown):
            lines.append(f"- ... and {len(failures) - len(shown)} more failure(s)")
        return "Fix the failing tests and finish the goal:\n" + "\n".join(lines)

    def _failed_dependency(self, plan: Plan, task: TaskModel) -> str | None:
        """Return the id of the first dependency of ``task`` that did not succeed."""
        for dependency in _dependency_ids(task):
            related = plan.task_by_id(dependency)
            if related is None:
                continue
            if _status(related) in (_FAILED, _BLOCKED):
                return dependency
        return None

    def _parallel_cap(self) -> int:
        """Return the worker concurrency cap; ``0`` or less means unlimited."""
        value = self._int_setting("max_parallel_tasks", _DEFAULT_MAX_PARALLEL_TASKS)
        return value if value > 0 else _UNLIMITED_TASKS

    def _cycle_cap(self) -> int:
        """Return the cycle limit; ``0`` means unlimited."""
        return self._int_setting("max_cycles", _DEFAULT_MAX_CYCLES)

    def _int_setting(self, name: str, default: int) -> int:
        """Read an integer setting, falling back to ``default`` when unusable."""
        value = getattr(self.settings, name, default)
        if value is None or isinstance(value, bool):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _emit(self, name: str, **payload: Any) -> None:
        """Render one event to the console and hand it to ``on_event``.

        The consumer call is wrapped in ``try``/``except`` so a raising or
        cancelling subscriber can never interrupt the loop.

        Args:
            name: One of :data:`EVENT_NAMES`.
            **payload: The event's payload, forwarded verbatim to the consumer.
        """
        _print(_render(name, payload))
        callback = self.on_event
        if callback is None:
            return
        try:
            callback(name, payload)
        except Exception as exc:
            detail = str(exc)[:_MAX_ERROR_CHARS]
            log.warning("event_consumer_failed", event_name=name, error=detail)

    def _emit_cycle_end(self, cycle: int, done: bool, reason: str) -> None:
        """Emit the closing ``cycle_end`` event for a cycle."""
        self._emit("cycle_end", cycle=cycle, done=bool(done), reason=reason)

    def _halt(self, reason: str, *, done: bool) -> None:
        """Record the terminal state and emit ``halt`` exactly once."""
        self.done = bool(done)
        self.halt_reason = str(reason)
        if self._halted:
            return
        self._halted = True
        self._emit("halt", cycles=int(self.cycles), reason=str(reason), done=bool(done))

    def _fail(self, cycle: int, exc: BaseException) -> None:
        """Log a cycle-level failure and emit the ``error`` event for it."""
        message = _error_text(exc)
        log.error("cycle_error", cycle=cycle, error=message)
        self._emit("error", cycle=cycle, message=message)

    def _print_banner(self) -> None:
        """Print the startup banner; it is decoration, not a loop event."""
        body = f"[bold green]Furrow[/bold green]\nGoal: {escape(self.goal)}"
        _print(Panel.fit(body, title="Furrow"))


def _resolve_settings(client: LLMClient | None, settings: Settings | None) -> Settings:
    """Pick the settings an orchestrator and its agents must use.

    An explicit ``settings`` wins; otherwise the client's own settings are used
    when they look like :class:`~furrow.config.Settings`, and the module
    singleton is the last resort.
    """
    if settings is not None:
        return settings
    candidate = getattr(client, "settings", None)
    if candidate is None or not hasattr(candidate, "max_cycles"):
        return default_settings
    return candidate


def _render(name: str, payload: dict[str, Any]) -> Any:
    """Return the console renderable for one event."""
    if name == "cycle_start":
        return f"[bold cyan]═══ Cycle {payload.get('cycle')} ═══[/bold cyan]"
    if name == "plan":
        return _render_plan(payload)
    if name == "task_start":
        return (
            f"[dim]→ {escape(str(payload.get('task_id')))}: "
            f"{escape(str(payload.get('description')))}[/dim]"
        )
    if name == "task_complete":
        return f"[green]✓ {escape(str(payload.get('task_id')))}[/green]"
    if name == "task_failed":
        return (
            f"[red]✗ {escape(str(payload.get('task_id')))}: "
            f"{escape(str(payload.get('error')))}[/red]"
        )
    if name == "task_blocked":
        return (
            f"[yellow]⊘ {escape(str(payload.get('task_id')))}: "
            f"{escape(str(payload.get('reason')))}[/yellow]"
        )
    if name == "tests":
        return _render_tests(payload)
    if name == "cycle_end":
        done = bool(payload.get("done"))
        style = "bold green" if done else "bold"
        return (
            f"[{style}]── cycle {payload.get('cycle')} end: "
            f"{escape(str(payload.get('reason')))} (done={done})[/{style}]"
        )
    if name == "error":
        return f"[bold red]! {escape(str(payload.get('message')))}[/bold red]"
    if name == "halt":
        return (
            f"[bold yellow]■ halted after {payload.get('cycles')} cycle(s): "
            f"{escape(str(payload.get('reason')))} "
            f"(done={bool(payload.get('done'))})[/bold yellow]"
    )
    return f"[dim]{name}[/dim]"


def _render_plan(payload: dict[str, Any]) -> Any:
    """Render the ``plan`` event, keeping JSON primitives visible."""
    plan = payload.get("plan")
    tasks = plan.get("tasks") if isinstance(plan, dict) else None
    if isinstance(tasks, list) and not tasks:
        return (
            "[yellow]No tasks planned: the planner considers the goal "
            f"{escape(str(payload.get('rationale')) or 'complete')}.[/yellow]"
        )
    return Panel(Pretty(plan), title="Plan", border_style="blue")


def _render_tests(payload: dict[str, Any]) -> str:
    """Render the ``tests`` event plus a bounded list of failures."""
    summary = escape(str(payload.get("summary")))
    failures = payload.get("failures") or []
    if payload.get("passed"):
        return f"[green]Tests passed: {summary}[/green]"
    lines = [f"[red]Tests failed: {summary}[/red]"]
    for item in list(failures)[:_MAX_PRINTED_FAILURES]:
        lines.append(f"  • {escape(str(item))}")
    if len(failures) > _MAX_PRINTED_FAILURES:
        lines.append(f"  • ... and {len(failures) - _MAX_PRINTED_FAILURES} more")
    return "\n".join(lines)


def _print(renderable: Any) -> None:
    """Print to the shared console, tolerating a broken terminal."""
    try:
        console.print(renderable)
    except Exception as exc:  # pragma: no cover - rendering must never break a run
        log.debug("console_render_failed", error=str(exc)[:_MAX_ERROR_CHARS])


def _status(task: TaskModel) -> str:
    """Return a task's status as a normalised lowercase string."""
    raw = getattr(task, "status", "")
    if isinstance(raw, Enum):
        raw = raw.value
    return str(raw).strip().lower()


def _dependency_ids(task: TaskModel) -> list[str]:
    """Return the task's declared dependency ids as strings."""
    return [str(item) for item in getattr(task, "dependencies", None) or []]


def _error_text(exc: BaseException) -> str:
    """Render an exception as a bounded, non-empty message."""
    text = str(exc).strip()
    if not text:
        text = type(exc).__name__
    return text[:_MAX_ERROR_CHARS]


def _optional_text(value: Any) -> str | None:
    """Return ``value`` as a string, or ``None`` when it is absent."""
    return None if value is None else str(value)
