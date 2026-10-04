"""Orchestrator: the engine behind Furrow's parallel development loop.

One *cycle* is plan -> execute -> test:

1. ``PlannerAgent`` turns the current focus into a ``Plan`` of tasks.
2. The plan is split into topological levels and each level is executed
   concurrently, never more than ``settings.max_parallel_tasks`` workers in
   flight. A task only starts once every dependency it declares is COMPLETED;
   a task whose dependency FAILED is skipped and marked FAILED, never run.
3. ``TesterAgent`` runs the workspace's test suite and judges it.

A planner failure that is still within ``MAX_CONSECUTIVE_PLANNER_FAILURES``
produces no plan, and therefore no cycle: nothing is executed and the suite is
**not** run, because a suite run against a plan that does not exist judges
nothing while costing a real subprocess run plus a verifier call. Such a cycle
goes straight to the cycle-end bookkeeping and then the next one.

The loop repeats until exactly one termination condition fires. The reason is
always recorded in ``self.stopped_reason``:

* ``goal complete`` -- every planned task applied edits and the suite passed.
* ``max_cycles`` -- ``settings.max_cycles`` cycles have run (0 means unlimited).
* ``stalled`` -- no file changed and the test verdict repeated for as many
  consecutive cycles as ``max_stalled_cycles``, so another cycle would only burn
  tokens. The halt fires one cycle *after* the threshold, because the first
  no-change cycle has no previous verdict to compare against; the message
  reports the real streak length, so it always matches the cycles that ran.
* ``no detectable test runner`` -- the tester could not verify anything. That is
  an unverifiable workspace, not a passing one, so the loop never claims success.
* ``planner failed`` -- no usable plan for ``MAX_CONSECUTIVE_PLANNER_FAILURES``
  cycles in a row.
* ``planner returned no tasks`` -- there is nothing left to decompose.
* ``cancelled`` -- ``asyncio`` cancellation; recorded, then re-raised.

``self.goal`` is the user's objective and is never mutated. ``self.focus`` is the
per-cycle objective handed to the planner: after a failing cycle it is the goal
plus the current failures. ``run()`` returns None; inspect ``self.tasks``,
``self.cycles``, ``self.test_results`` and ``self.stopped_reason`` for the
outcome, or pass ``on_event`` to observe the loop while it runs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TaskStatus, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()
logger = logging.getLogger(__name__)

#: Consecutive no-progress cycles tolerated before the loop gives up.
DEFAULT_MAX_STALLED_CYCLES = 3
#: Consecutive unusable plans tolerated before the loop gives up.
MAX_CONSECUTIVE_PLANNER_FAILURES = 3
#: Failures pasted into the next cycle's focus; the rest are summarised.
MAX_FOCUS_FAILURES = 20
#: File names printed per task before the list is elided.
MAX_LISTED_FILES = 8
#: Prefix of the ``stopped_reason`` that means the run actually succeeded.
COMPLETE_PREFIX = "goal complete"


class _NoPlan:
    """Sentinel type for "the planner failed, but the failure is still tolerated".

    A tolerated planner failure and a plan with zero tasks used to look identical
    to ``_cycle`` -- both left ``_plan_cycle`` returning ``None``, meaning "keep
    looping" -- so a cycle with no plan at all fell through to the scheduler and
    then to ``TesterAgent``, burning a full suite run per failed cycle and
    recording a verdict for a plan that never existed. The sentinel keeps the two
    apart; see :meth:`Orchestrator._skip_cycle`.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<no plan>"


#: The only instance of :class:`_NoPlan`.
_NO_PLAN = _NoPlan()


class Orchestrator:
    """Runs plan/execute/test cycles until the goal is met or the loop must stop."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        max_stalled_cycles: int = DEFAULT_MAX_STALLED_CYCLES,
    ) -> None:
        self.goal = goal
        self.focus = goal
        self.client = client or LLMClient()
        self.on_event = on_event
        self.max_stalled_cycles = max_stalled_cycles
        self.planner = PlannerAgent(client=self.client)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.test_results: list[TestResult] = []
        self.changed_files: list[str] = []
        self.stopped_reason: str | None = None
        self._by_id: dict[str, TaskModel] = {}
        self._planned_changes: list[str] = []
        self._changed_this_cycle = False
        self._planner_failures = 0
        self._stalled = 0
        self._no_progress = 0
        self._last_verdict: tuple[bool, tuple[str, ...]] | None = None

    @property
    def _settings(self) -> Settings:
        return self.client.settings

    async def run(self) -> None:
        """Drive the loop to completion, then record why it stopped in ``stopped_reason``."""
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        try:
            while True:
                limit = self._settings.max_cycles
                if limit and self.cycles >= limit:
                    self._halt(f"reached the max_cycles limit of {limit} cycle(s)")
                    return
                self.cycles += 1
                self._emit("cycle_start", {"cycle": self.cycles, "focus": self.focus})
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                console.print(f"[dim]Focus: {self._preview(self.focus)}[/dim]")
                reason = await self._cycle()
                self._emit(
                    "cycle_end",
                    {
                        "cycle": self.cycles,
                        "changed_files": list(self._planned_changes),
                        "stalled_cycles": self._stalled,
                    },
                )
                if reason is not None:
                    self._halt(reason)
                    return
        except asyncio.CancelledError:
            self._halt("cancelled before the goal was met")
            raise
        except Exception as exc:  # noqa: BLE001 - run() must not leak a raw traceback
            self._halt(f"unexpected orchestrator error: {type(exc).__name__}: {exc}")

    async def _cycle(self) -> str | None:
        """Run one cycle. Returns the halt reason, or None to keep looping."""
        self.tasks = []
        self._changed_this_cycle = False
        self._planned_changes = []

        planned = await self._plan_cycle()
        if planned is _NO_PLAN:
            return self._skip_cycle()
        if planned is not None:
            return planned

        gate = asyncio.Semaphore(self._settings.max_parallel_tasks)
        try:
            levels = self._dependency_levels(self.tasks)
        except ValueError as exc:
            console.print(f"[bold red]Unusable plan: {exc}[/bold red]")
            return f"the plan could not be scheduled: {exc}"

        for index, level in enumerate(levels, start=1):
            console.print(
                f"[bold magenta]Level {index}/{len(levels)}:[/bold magenta] "
                + ", ".join(task.id for task in level)
            )
            await asyncio.gather(*(self._run_task(task, gate) for task in level))

        return await self._test_cycle()

    def _skip_cycle(self) -> None:
        """Close out a cycle that produced no plan, and return no halt reason.

        There is nothing to execute and nothing new to verify. Before this path
        existed the loop scheduled an empty task list and still ran the whole
        suite, so a planner that failed on every cycle cost a real subprocess run
        plus a verifier call each time and filled ``test_results`` with verdicts
        about cycles that produced no plan at all.

        The stall counter is deliberately *not* touched. ``_stalled`` and
        ``_last_verdict`` only ever describe cycles that were actually tested, so
        an untested cycle is neither proof of progress (it must not reset the
        streak) nor proof of no progress (it must not extend it) -- the honest
        streak length is simply unknown until a real verdict arrives. Meanwhile
        ``MAX_CONSECUTIVE_PLANNER_FAILURES`` already bounds this case, and while
        the planner keeps failing it is the strictly tighter of the two limits.
        """
        console.print("[yellow]No plan this cycle: skipping execution and testing.[/yellow]")

    async def _plan_cycle(self) -> str | _NoPlan | None:
        """Ask the planner for this cycle's tasks.

        Returns a halt reason to stop the run, the ``_NO_PLAN`` sentinel when the
        planner failed but the failure is still within tolerance (skip the cycle
        and keep looping), or ``None`` when a usable plan was produced.
        """
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.focus)
        except Exception as exc:  # noqa: BLE001 - a bad plan must not kill the process
            self._planner_failures += 1
            console.print(f"[bold red]Planner failed: {type(exc).__name__}: {exc}[/bold red]")
            self._emit(
                "planner_error",
                {"cycle": self.cycles, "error": str(exc), "consecutive": self._planner_failures},
            )
            if self._planner_failures >= MAX_CONSECUTIVE_PLANNER_FAILURES:
                return f"the planner failed {self._planner_failures} cycles in a row: {exc}"
            return _NO_PLAN
        self._planner_failures = 0

        self.tasks = list(plan.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            {
                "cycle": self.cycles,
                "tasks": [task.short() for task in self.tasks],
                "rationale": plan.rationale,
            },
        )
        if not self.tasks:
            console.print("[yellow]The planner produced no tasks.[/yellow]")
            return "the planner returned no tasks to execute"
        return None

    def _dependency_levels(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into dependency-ordered levels.

        Raises ``ValueError`` when the declared dependencies contain a cycle, so a
        self-blocking plan is reported instead of deadlocking the loop.
        """
        self._by_id = {}
        duplicates: set[str] = set()
        for task in tasks:
            if task.id in self._by_id:
                duplicates.add(task.id)
            self._by_id.setdefault(task.id, task)
        if duplicates:
            console.print(
                f"[yellow]Warning: duplicate task ids {sorted(duplicates)}; "
                "dependencies on them are ambiguous.[/yellow]"
            )
        known = set(self._by_id)
        for task in tasks:
            for dep in task.dependencies:
                if dep not in known:
                    console.print(
                        f"[yellow]Warning: task {task.id} depends on unknown task {dep}; "
                        "the constraint is dropped.[/yellow]"
                    )

        pending = list(tasks)
        resolved: set[str] = set()
        levels: list[list[TaskModel]] = []
        while pending:
            ready = [
                task
                for task in pending
                if all(dep in resolved for dep in task.dependencies if dep in known)
            ]
            if not ready:
                stuck = sorted({task.id for task in pending})
                raise ValueError(f"dependency cycle between tasks {stuck}")
            levels.append(ready)
            resolved.update(task.id for task in ready)
            dispatched = {id(task) for task in ready}
            pending = [task for task in pending if id(task) not in dispatched]
        return levels

    async def _run_task(self, task: TaskModel, gate: asyncio.Semaphore) -> None:
        """Run one worker under the concurrency gate. Never raises."""
        async with gate:
            blocked = [dep for dep in task.dependencies if not self._dependency_ok(dep)]
            if blocked:
                task.status = TaskStatus.FAILED
                task.result = f"skipped: dependency {', '.join(blocked)} did not complete"
                console.print(f"  [bold red]✗[/bold red] {task.short()} — {task.result}")
                self._emit("task_done", self._task_event(task, skipped=True))
                return

            console.print(f"  [dim]→ {task.short()}[/dim]")
            self._emit("task_start", {"cycle": self.cycles, "task": task.id})
            worker = self._build_worker(task)
            try:
                summary = await worker.run()
            except Exception as exc:  # noqa: BLE001 - one task must not sink the cycle
                task.status = TaskStatus.FAILED
                task.result = f"worker raised {type(exc).__name__}: {exc}"
                console.print(f"  [bold red]✗[/bold red] {task.short()} — {task.result}")
                self._emit("task_done", self._task_event(task))
                return

            task.result = summary
            if worker.changed_files:
                task.status = TaskStatus.COMPLETED
                self._changed_this_cycle = True
                self._planned_changes.extend(worker.changed_files)
                for path in worker.changed_files:
                    if path not in self.changed_files:
                        self.changed_files.append(path)
                console.print(
                    f"  [green]✓[/green] {task.id}: "
                    f"{self._preview(self._list(worker.changed_files))}"
                )
            else:
                reason = worker.error or "the worker applied no edit to the workspace"
                task.status = TaskStatus.FAILED
                task.result = f"{reason}\n\n{summary}".strip()
                console.print(f"  [bold red]✗[/bold red] {task.short()} — {reason}")
            self._emit("task_done", self._task_event(task))

    def _build_worker(self, task: TaskModel) -> WorkerAgent:
        """Construct the worker for ``task``, with repair context only when there is one.

        The current failures are handed over only on a repair cycle (see
        :meth:`_repair_failures`). Without them the construction is exactly the
        historical ``WorkerAgent(task=..., client=...)``, so a first cycle and any
        green cycle keep getting the plain worker prompt.
        """
        failures = self._repair_failures()
        if not failures:
            return WorkerAgent(task=task, client=self.client)
        return WorkerAgent(task=task, client=self.client, failures=failures)

    def _repair_failures(self) -> list[str]:
        """The failures to repair this cycle; empty unless this is a repair cycle.

        A cycle is a repair cycle exactly when the previous cycle's test run produced
        failures -- the same condition that makes :meth:`_test_cycle` announce "Will
        attempt a repair cycle" and rebuild ``self.focus`` from :meth:`_focus_failures`.
        It is read from ``self.test_results``, which already holds that verdict, so no
        parallel bookkeeping is needed. An unverified run is not a repair signal: it
        carries no usable failure text and halts the loop instead.
        """
        if not self.test_results:
            return []
        last = self.test_results[-1]
        if last.verified and not last.passed:
            return list(last.failures)
        return []

    def _dependency_ok(self, dep: str) -> bool:
        """True when ``dep`` refers to a known task that finished COMPLETED."""
        task = self._by_id.get(dep)
        return task is not None and task.status is TaskStatus.COMPLETED

    async def _test_cycle(self) -> str | None:
        """Run the test suite and turn the verdict into loop control."""
        try:
            with Status("[bold yellow]Testing...", console=console):
                result = await TesterAgent(client=self.client).run(self.goal, self.tasks)
        except Exception as exc:  # noqa: BLE001 - infrastructure loss is not a test failure
            console.print(f"[bold red]Tester failed: {type(exc).__name__}: {exc}[/bold red]")
            return f"the tester could not run: {type(exc).__name__}: {exc}"

        self.test_results.append(result)
        self._emit(
            "test",
            {
                "cycle": self.cycles,
                "passed": result.passed,
                "verified": result.verified,
                "summary": result.summary,
                "failures": list(result.failures),
            },
        )

        if not result.verified:
            console.print(f"[bold red]Unverifiable workspace: {result.summary}[/bold red]")
            console.print(
                "[yellow]No test runner was detected, so this result carries no signal. "
                "Furrow is halting instead of claiming an unproven win.[/yellow]"
            )
            return (
                "no detectable test runner: the workspace is unverifiable, not passing "
                f"({result.summary})"
            )

        if result.passed:
            console.print(f"[green]Tests passed: {result.summary}[/green]")
        else:
            console.print(f"[bold red]Tests failed:[/bold red] {result.summary}")
            for failure in result.failures:
                console.print(f"  • {failure}")

        verdict = (result.passed, tuple(result.failures))
        # Off-by-one source of the old, misleading halt message: the first
        # no-change cycle compares against ``_last_verdict is None``, so it
        # *resets* ``_stalled`` to 0 instead of incrementing it and the halt
        # lands one cycle after ``max_stalled_cycles``. That threshold behaviour
        # is kept as-is, but ``_no_progress`` counts the real streak of cycles
        # that changed no file and repeated the verdict (the verdict-less first
        # cycle included), so the message can be derived from the state instead
        # of hardcoding a number that miscounts the cycles that actually ran.
        repeated = self._last_verdict is not None and verdict == self._last_verdict
        if self._changed_this_cycle or (self._last_verdict is not None and not repeated):
            self._no_progress = 0
        else:
            self._no_progress += 1
        if self._changed_this_cycle or not repeated:
            self._stalled = 0
        else:
            self._stalled += 1
        self._last_verdict = verdict
        if self._stalled >= self.max_stalled_cycles:
            console.print(f"[bold red]No progress for {self._no_progress} cycles.[/bold red]")
            return (
                f"stalled: {self._no_progress} consecutive cycles changed no file and "
                "produced identical test failures"
            )

        if result.passed and self._is_done():
            return f"{COMPLETE_PREFIX}: every planned task applied its edits and tests pass"
        if result.passed:
            console.print(
                "[yellow]Tests pass but some planned tasks are still not applied.[/yellow]"
            )
            self.focus = self._focus_incomplete()
        else:
            console.print("[yellow]Will attempt a repair cycle.[/yellow]")
            self.focus = self._focus_failures(result.failures)
        return None

    def _focus_failures(self, failures: list[str]) -> str:
        """Build the next cycle's focus: the untouched goal plus the current failures."""
        shown = failures[:MAX_FOCUS_FAILURES]
        body = "\n".join(f"  - {line}" for line in shown)
        omitted = len(failures) - len(shown)
        if omitted:
            body += f"\n  - ... and {omitted} further failure(s)"
        return f"{self.goal}\n\nThe test suite is failing. Fix these failures:\n{body}"

    def _focus_incomplete(self) -> str:
        """Build the next cycle's focus for tasks whose edits never landed."""
        missing = [task.id for task in self.tasks if task.status is not TaskStatus.COMPLETED]
        return (
            f"{self.goal}\n\nThe test suite passes but these planned tasks applied "
            f"no usable edit and must still land: {', '.join(missing)}"
        )

    def _is_done(self) -> bool:
        """True when the latest plan fully applied and the latest verified run passed."""
        if not self.tasks or not self.test_results:
            return False
        if any(task.status is not TaskStatus.COMPLETED for task in self.tasks):
            return False
        last = self.test_results[-1]
        return bool(last.verified and last.passed)

    def _get_tasks(self) -> list[TaskModel]:
        """The tasks of the most recent plan; empty until the first plan lands."""
        return list(self.tasks)

    def _task_event(self, task: TaskModel, *, skipped: bool = False) -> dict[str, Any]:
        return {
            "cycle": self.cycles,
            "task": task.id,
            "status": task.status.value,
            "skipped": skipped,
            "result": task.result,
        }

    def _halt(self, reason: str) -> None:
        """Record, report and broadcast why the loop stopped."""
        self.stopped_reason = reason
        style = "bold green" if reason.startswith(COMPLETE_PREFIX) else "bold yellow"
        console.print(f"[{style}]Halting: {reason}[/{style}]")
        self._emit(
            "halt",
            {
                "reason": reason,
                "cycles": self.cycles,
                "changed_files": list(self.changed_files),
                "complete": reason.startswith(COMPLETE_PREFIX),
            },
        )

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        """Notify the optional progress hook; a broken callback never breaks the loop."""
        if self.on_event is None:
            return
        try:
            self.on_event(event, payload)
        except Exception:
            logger.debug("progress hook failed on event %s", event, exc_info=True)

    @staticmethod
    def _list(paths: list[str]) -> str:
        """Comma-joined paths, elided past ``MAX_LISTED_FILES``."""
        shown = ", ".join(paths[:MAX_LISTED_FILES])
        if len(paths) > MAX_LISTED_FILES:
            shown += f" (+{len(paths) - MAX_LISTED_FILES} more)"
        return shown

    @staticmethod
    def _preview(text: str, limit: int = 200) -> str:
        """Single-line, length-capped rendering of a long message."""
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[:limit] + " ..."
