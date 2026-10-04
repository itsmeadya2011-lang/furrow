"""The Furrow orchestrator: an autonomous plan/execute/test development loop.

Real state, not a single counter, drives the loop:

* :attr:`Orchestrator.tasks` holds the tasks of the most recent plan and is
  refreshed on every cycle, so progress is observable through
  :attr:`Orchestrator.state`.
* :attr:`Orchestrator.original_goal` is written once in ``__init__`` and never
  mutated.  A failing test suite is folded into a follow-up *prompt* for the
  planner instead of overwriting the goal the user asked for.
* The loop stops on the first of: a stop request, a completed goal, a plan with
  no tasks at all, or an exhausted ``max_cycles`` budget (``0`` means forever).

Tasks are scheduled in dependency order: the plan is split into topological
levels and each level runs in parallel, bounded by
``asyncio.Semaphore(max_parallel_tasks)``.  Unknown dependency ids and
dependency cycles are reported as task failures instead of hanging.
"""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

console = Console()

#: Signature of the progress-event sink handed to ``Orchestrator(on_event=...)``.
EventCallback = Callable[[dict[str, Any]], None]

#: The task lifecycle a plan moves through.
_TaskState: TypeAlias = Literal["pending", "in_progress", "completed", "failed"]

#: Every value :attr:`Orchestrator.stopped_reason` can take.
REASON_GOAL_COMPLETE = "goal_complete"
REASON_BUDGET = "cycle_budget_exhausted"
REASON_STOPPED = "stop_requested"
REASON_NO_TASKS = "no_tasks_planned"
REASON_ERROR = "error"


def _as_text(value: Any) -> str:
    """Coerce any worker return value into the ``str`` a task result holds.

    ``str()`` on an exception object loses the type, so exceptions (including
    ``BaseException``) are rendered with their traceback instead.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseException):
        summary = f"{type(value).__name__}: {value}".rstrip()
        rendered = "".join(traceback.format_exception(value)).strip()
        return f"{summary}\n{rendered}" if rendered else summary
    return str(value)


class Orchestrator:
    """Runs the plan -> work -> test loop for a single goal."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        on_event: EventCallback | None = None,
    ) -> None:
        # The goal exactly as requested; never mutated by the loop.
        self.original_goal = goal
        # The planning prompt for the current cycle: goal + follow-up context.
        self.goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = getattr(self.client, "settings", None) or Settings()
        self.planner = PlannerAgent(client=self.client)

        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.stopped_reason: str | None = None

        self._on_event = on_event
        self._plan: Plan | None = None
        self._running = False
        self._tests_passed = False
        self._feedback: str | None = None
        self._last_test: dict[str, Any] | None = None
        self._errors: list[str] = []
        self._emit_errors = 0
        self._stop_event = asyncio.Event()
        self._stop_requested = False
        self._stop_emitted = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._max_parallel = max(1, int(self.settings.max_parallel_tasks))
        self._semaphore = asyncio.Semaphore(self._max_parallel)

    # ------------------------------------------------------------------ events

    def _emit(self, kind: str, **payload: Any) -> None:
        """Deliver a progress event to the callback; never raises into the loop."""
        if self._on_event is None:
            return
        event: dict[str, Any] = {"kind": kind, "cycle": self.cycles, **payload}
        try:
            self._on_event(event)
        except Exception:
            # A broken listener must never take the run down with it.
            self._emit_errors += 1

    def request_stop(self) -> None:
        """Ask the run to finish the current work and then stop.

        Safe to call from another task or from a thread: the flag is a plain
        boolean and the event is set through ``call_soon_threadsafe`` whenever a
        loop is running.
        """
        self._stop_requested = True
        loop = self._loop
        if loop is not None and not loop.is_closed() and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._stop_event.set)
                return
            except RuntimeError:  # loop closed underneath us
                pass
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """True while :meth:`run` is executing."""
        return self._running

    @property
    def state(self) -> dict[str, Any]:
        """A JSON-serializable snapshot of the run."""
        return {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "running": self._running,
            "stopped_reason": self.stopped_reason,
            "tasks": [task.model_dump() for task in self.tasks],
            "completed": self._count("completed"),
            "failed": self._count("failed"),
            "unfinished": sum(
                1
                for task in self.tasks
                if task.status not in ("completed", "failed")
            ),
            "max_cycles": int(self.settings.max_cycles),
            "max_parallel_tasks": self._max_parallel,
            "tests_passed": self._tests_passed,
            "feedback": self._feedback,
            "last_test_result": self._last_test,
            "errors": list(self._errors),
        }

    # -------------------------------------------------------------- main loop

    async def run(self) -> dict[str, Any]:
        """Run cycles until the goal is met, the budget runs out, or a stop lands."""
        self._running = True
        self._loop = asyncio.get_running_loop()
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        self._emit("run_start", goal=self.original_goal)
        try:
            reason = await self._loop_until_done()
        except asyncio.CancelledError:
            self.stopped_reason = REASON_STOPPED
            raise
        finally:
            self._running = False
        self.stopped_reason = reason
        completed = self._count("completed")
        failed = self._count("failed")
        console.print(
            f"[bold]Furrow finished[/bold] ({reason}): "
            f"{self.cycles} cycle(s), {completed} completed, {failed} failed."
        )
        self._emit(
            "run_end",
            reason=reason,
            cycles=self.cycles,
            completed=completed,
            failed=failed,
        )
        return self.state

    async def _loop_until_done(self) -> str:
        """Drive cycles until a stop condition fires; return that reason."""
        reason = REASON_ERROR
        while True:
            if self._check_stop():
                return REASON_STOPPED
            if self._budget_exhausted():
                return REASON_BUDGET
            self.cycles += 1
            self._emit("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]=== Cycle {self.cycles} ===[/bold cyan]")
            try:
                await self._cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._tests_passed = False
                self._record_error(f"cycle {self.cycles}: {_as_text(exc)}")
            self._emit("cycle_end", cycle=self.cycles)
            if self._is_done():
                reason = self._exit_reason() or REASON_ERROR
                break
        return reason

    async def _cycle(self) -> None:
        """Plan, execute (dependency ordered), then test one slice of the work."""
        if self._check_stop():
            return

        self.goal = self._planner_goal()
        plan = await self._plan_next()
        if plan is None:
            self.tasks = []
            self._tests_passed = False
            return

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            tasks=[task.model_dump() for task in plan.tasks],
            rationale=plan.rationale,
        )
        self._plan = plan
        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._tests_passed = False
            return

        levels, dependency_error = self._dependency_levels(plan.tasks)
        if dependency_error is not None:
            console.print(f"[red]Dependency error: {dependency_error}[/red]")
            self._record_error(dependency_error)
            self._fail_tasks(plan.tasks, dependency_error)
            self._tests_passed = False
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            await self._execute(levels)
        if self._check_stop():
            return

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.original_goal, self.tasks
            )

        self._tests_passed = test_result.passed
        self._last_test = test_result.model_dump()
        self._emit(
            "test_result",
            passed=test_result.passed,
            summary=test_result.summary,
            failures=list(test_result.failures),
        )
        if test_result.passed:
            self._feedback = None
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  - {failure}")
        console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        self._feedback = self._failure_context(test_result.summary, test_result.failures)

    async def _plan_next(self) -> Plan | None:
        """Ask the planner for the next plan; ``None`` when planning fails."""
        try:
            with Status("[bold yellow]Planning...", console=console):
                return await self.planner.plan(self.goal)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = f"planning failed: {_as_text(exc)}"
            console.print(f"[red]{detail}[/red]")
            self._record_error(detail)
            self._emit("plan", tasks=[], rationale=detail)
            return None

    def _planner_goal(self) -> str:
        """The original goal, plus the previous cycle's failure context."""
        if not self._feedback:
            return self.original_goal
        return (
            f"{self.original_goal}\n\n"
            f"Previous cycle feedback (cycle {self.cycles}):\n{self._feedback}\n\n"
            "Plan the next slice of work that resolves the problems above."
        )

    def _failure_context(self, summary: str, failures: list[str]) -> str:
        """Render a follow-up prompt body from a failing test result."""
        detail = "\n".join(f"- {line}" for line in failures if line.strip())
        if not detail:
            detail = summary
        return f"Fix failing tests: {summary}\n{detail}"

    # -------------------------------------------------------------- execution

    async def _execute(self, levels: list[list[TaskModel]]) -> None:
        """Run every dependency level in order, each level in parallel."""
        for level in levels:
            if self._check_stop():
                return
            results = await asyncio.gather(
                *(self._run_task(task) for task in level), return_exceptions=True
            )
            for result in results:
                # Cancellation is not a task failure: let it unwind the run.
                if isinstance(result, asyncio.CancelledError):
                    raise result

    async def _run_task(self, task: TaskModel) -> None:
        """Run one task under the concurrency cap and record its outcome."""
        if self._check_stop():
            return
        async with self._semaphore:
            if self._check_stop():
                return
            task.status = "in_progress"
            self._emit("task_start", task_id=task.id, description=task.description)
            console.print(f"[cyan]Task {task.id}: {task.description}[/cyan]")
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
            except BaseException as exc:
                self._finish_task(task, "failed", _as_text(exc))
                if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                    raise
            else:
                self._finish_task(task, "completed", _as_text(result))

    def _finish_task(self, task: TaskModel, status: _TaskState, result: str) -> None:
        """Store a task outcome and announce it."""
        task.status = status
        task.result = result
        first_line = result.strip().splitlines()[0] if result.strip() else ""
        if status == "completed":
            console.print(f"[green]Task {task.id} completed[/green]")
        else:
            console.print(f"[red]Task {task.id} failed: {first_line}[/red]")
        self._emit("task_end", task_id=task.id, status=status, result=result)

    def _fail_tasks(self, tasks: list[TaskModel], error: str) -> None:
        """Fail every still-pending task of an unusable plan."""
        for task in tasks:
            if task.status == "pending":
                self._finish_task(task, "failed", error)

    def _dependency_levels(
        self, tasks: list[TaskModel]
    ) -> tuple[list[list[TaskModel]], str | None]:
        """Group tasks into topological levels.

        Returns ``(levels, None)`` on success or ``([], error)`` when the
        dependency graph is unusable (duplicate ids, unknown ids, or a cycle).
        No input can make this loop forever: each pass resolves at least one
        task or reports the unresolvable remainder as a cycle.
        """
        by_id = {task.id: task for task in tasks}
        seen: set[str] = set()
        duplicates: set[str] = set()
        for task in tasks:
            if task.id in seen:
                duplicates.add(task.id)
            seen.add(task.id)
        if duplicates:
            return [], f"plan has duplicate task ids: {', '.join(sorted(duplicates))}"

        for task in tasks:
            unknown = [dep for dep in task.dependencies if dep not in by_id]
            if unknown:
                return [], (
                    f"task {task.id} depends on unknown task id(s): {', '.join(unknown)}"
                )

        pending = {task.id: set(task.dependencies) for task in tasks}
        resolved: set[str] = set()
        levels: list[list[TaskModel]] = []
        while pending:
            ready = [task_id for task_id, deps in pending.items() if deps <= resolved]
            if not ready:
                return [], f"dependency cycle detected among tasks: {', '.join(sorted(pending))}"
            levels.append([by_id[task_id] for task_id in ready])
            resolved.update(ready)
            for task_id in ready:
                del pending[task_id]
        return levels, None

    # ------------------------------------------------------------ termination

    def _check_stop(self) -> bool:
        """Observe a stop request; emits ``stop_requested`` exactly once."""
        if not self._stop_requested:
            return False
        if not self._stop_emitted:
            self._stop_emitted = True
            self._emit("stop_requested")
        return True

    def _budget_exhausted(self) -> bool:
        """True when ``max_cycles > 0`` and that many cycles have already run."""
        max_cycles = int(self.settings.max_cycles)
        return max_cycles > 0 and self.cycles >= max_cycles

    def _exit_reason(self) -> str | None:
        """The reason to stop after the current cycle, or ``None`` to keep going."""
        # _check_stop() also announces the request, so a stop that lands during a
        # cycle still emits ``stop_requested`` exactly once.
        if self._check_stop():
            return REASON_STOPPED
        if self.tasks and self._tests_passed and not self._count("failed"):
            return REASON_GOAL_COMPLETE
        if not self.tasks:
            return REASON_NO_TASKS
        if self._budget_exhausted():
            return REASON_BUDGET
        # Failed tasks or failing tests: plan another cycle to try to fix them.
        return None

    def _is_done(self) -> bool:
        """True when the loop should stop after the current cycle."""
        return self._exit_reason() is not None

    def _count(self, status: str) -> int:
        """Number of tasks in the current plan with ``status``."""
        return sum(1 for task in self.tasks if task.status == status)

    def _record_error(self, message: str) -> None:
        """Remember a cycle-level error for :attr:`state`."""
        self._errors.append(message)
