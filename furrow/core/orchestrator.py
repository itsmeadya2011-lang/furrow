"""The Furrow loop: plan, execute in dependency order, test, repeat.

This module owns no provider logic. It schedules :class:`PlannerAgent`,
:class:`WorkerAgent` and :class:`TesterAgent`, keeps the task state that has to
survive a cycle, and decides when to stop. Everything worth knowing about a run is
reported twice -- once to the console and once to the ``on_event`` callback -- so the
CLI and the web UI observe exactly the same run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, get_settings
from furrow.llm import LLMClient

#: Safety valve for the otherwise unbounded loop: a run stops after this many cycles in
#: a row have ended in an exception. Provider outages (429s, timeouts) and malformed model
#: output are the usual causes and both often clear on their own, so a short streak is
#: tolerated; a long one means the loop is going nowhere and burning API credits.
MAX_CONSECUTIVE_CYCLE_ERRORS = 3

#: How many times one task id may be handed to a worker before it is abandoned.
#: Retrying is what turns a transient provider error into a completed task; retrying
#: forever turns a task the model cannot do into an infinite loop.
MAX_TASK_ATTEMPTS = 2

#: Consecutive cycles the planner may answer with no tasks, while the tests are still
#: failing, before the run gives up. An empty plan with green tests means "nothing left
#: to do"; an empty plan with red tests means the loop is stuck, and three is short
#: enough to stop the spend and long enough to ride out one confused answer.
MAX_EMPTY_PLAN_STREAK = 3

_UNRESOLVED = 0
_ACTIVE = 1
_SETTLED = 2


class Orchestrator:
    """Run a goal through repeated plan/execute/test cycles.

    State that has to outlive a cycle -- task statuses, per-task attempt counts, the last
    test result -- lives on the instance. Nothing is recomputed from scratch each cycle,
    which is what lets cycle 2 know what cycle 1 actually achieved.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        settings: Settings | None = None,
        console: Console | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        # `goal` is the user's objective and is never reassigned. Failure text from a
        # cycle lives in `feedback` and is handed to the next planner/worker call, so
        # the objective survives the whole run instead of being overwritten by a
        # test-failure blob on cycle 1.
        self.goal = goal
        self.feedback = ""
        self.settings = settings if settings is not None else get_settings()
        # Only a client this object created may be closed by it: an injected client is
        # shared with whoever injected it and outlives this run.
        self._owns_client = client is None
        self.client = client if client is not None else LLMClient(settings=self.settings)
        # Injectable on purpose: a module-level Console under uvicorn interleaves
        # spinners with the server's own log lines, and tests need a StringIO sink.
        self.console = console if console is not None else Console()
        self._on_event = on_event
        self.planner = PlannerAgent(client=self.client, settings=self.settings)

        self._cycles = 0
        self._tasks: list[TaskModel] = []
        self._attempts: dict[str, int] = {}
        self._cancelled = False
        self._consecutive_errors = 0
        self._empty_plan_streak = 0
        self._test_result: TestResult | None = None
        self._complete = False
        self._cycle_tasks: list[TaskModel] = []
        self._planned_nothing = False

    @property
    def cycles(self) -> int:
        """Cycles started so far."""
        return self._cycles

    @property
    def tasks(self) -> list[TaskModel]:
        """Every task this run has planned, newest last (statuses are live)."""
        return list(self._tasks)

    @property
    def done(self) -> bool:
        """True only when the run finished because the goal was genuinely met."""
        return self._complete

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called."""
        return self._cancelled

    @property
    def last_test_result(self) -> TestResult | None:
        """The most recent tester verdict, or ``None`` before the first test run."""
        return self._test_result

    def cancel(self) -> None:
        """Ask the loop to stop at its next safe point.

        Cooperative on purpose: an in-flight worker is allowed to finish so it cannot
        leave a half-written file behind. The run then reports itself as cancelled, not
        complete.
        """
        if not self._cancelled:
            self._cancelled = True
            self._emit("warning", "Cancellation requested; stopping at the next safe point.")

    async def run(self) -> None:
        """Drive plan -> execute -> test cycles until one of these fires.

        Finished successfully (``run_done`` carries ``complete=True``):

        * every task of the cycle reached ``completed`` **and** the tester passed;
        * the planner returned no tasks and the tester passed -- nothing left to do; or
        * the tester passed while no failed task still has an attempt left, so another
          cycle could not change the verdict (reported with an explicit warning).

        Stopped without completion (``complete=False`` plus the reason):

        * :meth:`cancel` was called (reported as cancelled, never as complete);
        * ``settings.max_cycles`` was reached -- ``0`` means unlimited;
        * ``MAX_CONSECUTIVE_CYCLE_ERRORS`` cycles in a row raised (circuit breaker); or
        * ``MAX_EMPTY_PLAN_STREAK`` cycles in a row produced an empty plan while the
          tests were still failing (stagnation).

        Raises:
            asyncio.CancelledError: if the task running this coroutine is cancelled.
                That is a shutdown, not a failure: the ``finally`` still closes a
                self-owned client.
        """
        self.console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        self._emit("run_start", f"Starting Furrow: {self.goal}", goal=self.goal)
        reason = ""
        try:
            while True:
                if self._cancelled:
                    reason = "Cancelled on request."
                    break
                self._cycles += 1
                cycle = self._cycles
                self._emit("cycle_start", f"═══ Cycle {cycle} ═══", cycle=cycle)
                self.console.print(f"\n[bold cyan]═══ Cycle {cycle} ═══[/bold cyan]")
                try:
                    await self._cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # one bad cycle must not end the run
                    self._consecutive_errors += 1
                    detail = f"Cycle {cycle} failed: {type(exc).__name__}: {exc}"
                    self._emit("error", detail, cycle=cycle, error=detail)
                    self.console.print(f"[red]{detail}[/red]")
                    if self._consecutive_errors >= MAX_CONSECUTIVE_CYCLE_ERRORS:
                        reason = (
                            "Circuit breaker: stopped after "
                            f"{self._consecutive_errors} consecutive errored cycles "
                            f"(limit {MAX_CONSECUTIVE_CYCLE_ERRORS}). Last error: {detail}"
                        )
                        break
                    continue
                self._consecutive_errors = 0
                self._report_cycle_end(cycle)

                # Cancellation is checked before success: a cancel that lands during
                # the last cycle must never be reported as the goal being met.
                if self._cancelled:
                    reason = "Cancelled on request."
                    break
                completion = self._completion_reason()
                if completion is not None:
                    self._complete = True
                    reason = completion
                    break
                stop = self._stop_reason()
                if stop is not None:
                    reason = stop
                    break
        finally:
            await self._close_client()

        message = self._final_message(reason)
        self._emit(
            "run_done",
            message,
            complete=self._complete,
            cancelled=self._cancelled,
            cycles=self._cycles,
            reason=reason,
        )
        if self._complete:
            self.console.print(f"[bold green]Goal complete. {reason}[/bold green]")
        else:
            self.console.print(f"[yellow]Stopped. {reason}[/yellow]")

    # ------------------------------------------------------------------ agents

    async def _plan(self, feedback: str) -> Plan:
        """Ask the planner for this cycle's plan, with *feedback* attached."""
        with self._spinner("[bold yellow]Planning...[/bold yellow]"):
            return await self.planner.plan(self.goal, feedback=feedback)

    async def _test(self, tasks: list[TaskModel]) -> None:
        """Run the project suite and record the verdict as :attr:`last_test_result`."""
        with self._spinner("[bold yellow]Testing...[/bold yellow]"):
            self._test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, tasks
            )
        result = self._test_result
        assert result is not None  # the tester always assigns before returning
        message = (
            f"Tests passed: {result.summary}"
            if result.passed
            else f"Tests failed: {result.summary}"
        )
        self._emit(
            "tests",
            message,
            cycle=self._cycles,
            passed=result.passed,
            summary=result.summary,
            failures=list(result.failures),
        )
        if result.passed:
            self.console.print(f"[green]{message}[/green]")
            return
        self.console.print(f"[red]{message}[/red]")
        for failure in result.failures:
            self.console.print(f"  • {failure}")
        detail = "\n".join(result.failures) or result.summary
        self.console.print("[yellow]Will attempt a fix in the next cycle.[/yellow]")
        self._emit(
            "warning",
            "Tests are failing; the next cycle is planned with this failure output.",
            cycle=self._cycles,
            failures=list(result.failures),
        )
        self._add_feedback(f"Fix failing tests:\n{detail}")

    # ------------------------------------------------------------------- cycle

    async def _cycle(self) -> None:
        """Plan, execute and test exactly once.

        Raises whatever the agents raise; :meth:`run` turns that into a recorded cycle
        error so the loop survives it.
        """
        feedback, self.feedback = self.feedback, ""
        self._cycle_tasks = []
        self._planned_nothing = False

        retries = self._retry_queue()
        for task in retries:
            task.status = "pending"
        plan = await self._plan(feedback)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            self._plan_message(plan, retries),
            cycle=self._cycles,
            rationale=plan.rationale,
            tasks=[task.model_dump() for task in plan.tasks],
            retried=[task.id for task in retries],
        )

        batch = [*retries, *self._within_attempt_cap(plan, {t.id for t in retries})]
        self._tasks.extend(batch)
        self._cycle_tasks = batch
        self._planned_nothing = not batch
        if self._planned_nothing:
            self._empty_plan_streak += 1
            self._emit(
                "warning",
                f"Planner returned no work to do "
                f"(empty plan {self._empty_plan_streak} of {MAX_EMPTY_PLAN_STREAK}).",
                cycle=self._cycles,
                streak=self._empty_plan_streak,
            )
            self.console.print(
                f"[yellow]Nothing to do: empty plan "
                f"{self._empty_plan_streak}/{MAX_EMPTY_PLAN_STREAK}.[/yellow]"
            )
        else:
            self._empty_plan_streak = 0

        cycle_error = _dependency_cycle(batch)
        if cycle_error is not None:
            detail = "Plan has a dependency cycle: " + " -> ".join(cycle_error)
            self.console.print(f"[red]{detail}[/red]")
            self._emit("error", detail, cycle=self._cycles, cycle_path=cycle_error)
            self._add_feedback(detail)
            for task in batch:
                if task.status == "pending":
                    task.status = "failed"
                    task.result = "not started: the plan has a dependency cycle"
                    self._emit(
                        "task_failed",
                        f"Task {task.id} failed: dependency cycle in the plan",
                        cycle=self._cycles,
                        task=task.id,
                        error="dependency cycle",
                    )
        else:
            runnable = self._runnable_tasks(batch)
            if runnable:
                await self._execute(runnable, feedback)
        await self._test(batch)

    def _plan_message(self, plan: Plan, retries: Sequence[TaskModel]) -> str:
        """One line describing the plan, plus any retries carried into this cycle."""
        summary = ", ".join(task.id for task in plan.tasks) or "no tasks"
        message = f"Planned {len(plan.tasks)} task(s) for cycle {self._cycles}: {summary}"
        if retries:
            message += f" (retrying {', '.join(task.id for task in retries)})"
        return message

    # -------------------------------------------------------------- execution

    def _runnable_tasks(self, batch: Sequence[TaskModel]) -> list[TaskModel]:
        """Split *batch* into tasks that may run and tasks blocked by a failed dep.

        Only dependencies from *earlier* cycles can block a task here: ordering among
        tasks of the same plan is the scheduler's job. A task whose earlier-cycle
        dependency never completed is never launched -- running it would build on a
        result that does not exist -- and is left ``pending`` so it cannot be mistaken
        for work that was done.
        """
        by_id = {task.id: task for task in batch}
        runnable: list[TaskModel] = []
        for task in batch:
            blocker = ""
            for dep in task.dependencies:
                if dep in by_id:
                    continue
                status = self._status_of(dep, by_id)
                if status != "completed":
                    blocker = f"dependency {dep} is {status or 'unknown'}"
                    break
            if blocker:
                message = f"Task {task.id} not started: {blocker}"
                self.console.print(f"[yellow]{message}[/yellow]")
                self._emit("warning", message, cycle=self._cycles, task=task.id, dependency=blocker)
                self._add_feedback(message)
            else:
                runnable.append(task)
        return runnable

    async def _execute(self, tasks: Sequence[TaskModel], feedback: str) -> None:
        """Run *tasks* in topological layers, at most ``max_parallel_tasks`` at a time."""
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        by_id = {task.id: task for task in tasks}
        remaining = list(tasks)
        message = (
            f"Executing {len(tasks)} task(s) in dependency order "
            f"(at most {self.settings.max_parallel_tasks} in parallel)."
        )
        self.console.print(f"[dim]{message}[/dim]")
        while remaining:
            if self._cancelled:
                message = (
                    f"Cancelled: {len(remaining)} task(s) were never started "
                    f"({', '.join(task.id for task in remaining)})."
                )
                self.console.print(f"[yellow]{message}[/yellow]")
                self._emit(
                    "warning",
                    message,
                    cycle=self._cycles,
                    unstarted=[task.id for task in remaining],
                )
                return
            layer = [
                task
                for task in remaining
                if all(self._status_of(dep, by_id) == "completed" for dep in task.dependencies)
            ]
            if not layer:
                stuck = _dependency_cycle(remaining) or [task.id for task in remaining]
                detail = "Cannot schedule tasks: unsatisfiable dependencies in " + " -> ".join(
                    stuck
                )
                self.console.print(f"[red]{detail}[/red]")
                self._emit("error", detail, cycle=self._cycles, cycle_path=stuck)
                self._add_feedback(detail)
                for task in remaining:
                    task.status = "failed"
                    task.result = "not started: unsatisfiable dependencies"
                    self._emit(
                        "task_failed",
                        f"Task {task.id} failed: unsatisfiable dependencies",
                        cycle=self._cycles,
                        task=task.id,
                        error="unsatisfiable dependencies",
                    )
                return
            layer_ids = {task.id for task in layer}
            await self._run_layer(layer, semaphore, feedback)
            remaining = [task for task in remaining if task.id not in layer_ids]

    async def _run_layer(
        self,
        layer: Sequence[TaskModel],
        semaphore: asyncio.Semaphore,
        feedback: str,
    ) -> list[Any]:
        """Run one dependency layer concurrently and record each outcome."""

        async def run_one(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run(
                    feedback=feedback
                )

        for task in layer:
            task.status = "running"
            # The attempt is counted here, not inside the semaphore, so a task that is
            # queued behind the parallel cap is still known to have been started.
            self._attempts[task.id] = self._attempts.get(task.id, 0) + 1
            self.console.print(f"[cyan]Task {task.id} started:[/cyan] {task.description}")
            self._emit(
                "task_start",
                f"Task {task.id} started: {task.description}",
                cycle=self._cycles,
                task=task.id,
                description=task.description,
                attempt=self._attempts[task.id],
            )
        results = await asyncio.gather(*(run_one(task) for task in layer), return_exceptions=True)
        for task, result in zip(layer, results, strict=True):
            # BaseException, not Exception: a worker cancelled mid-flight is not a
            # success, and recording it as one is exactly how a run claims success
            # after every task was torn down.
            if isinstance(result, BaseException):
                detail = f"{type(result).__name__}: {result}"
                task.status = "failed"
                task.result = detail
                self.console.print(f"[red]Task {task.id} failed: {detail}[/red]")
                self._emit(
                    "task_failed",
                    f"Task {task.id} failed: {detail}",
                    cycle=self._cycles,
                    task=task.id,
                    error=detail,
                    attempt=self._attempts.get(task.id, 0),
                )
                self._add_feedback(f"Task {task.id} failed: {detail}")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                self._emit(
                    "task_done",
                    f"Task {task.id} completed.",
                    cycle=self._cycles,
                    task=task.id,
                    attempt=self._attempts.get(task.id, 0),
                    result=result,
                )
        return list(results)

    # --------------------------------------------------------------- feedback

    def _add_feedback(self, text: str) -> None:
        """Queue *text* as feedback for the next cycle's planner and workers."""
        text = text.strip()
        if not text:
            return
        self.feedback = f"{self.feedback}\n\n{text}" if self.feedback else text

    # ------------------------------------------------------------------ state

    def _retry_queue(self) -> list[TaskModel]:
        """Failed tasks from earlier cycles that still have an attempt left.

        Keyed by task id, newest first-wins: one failed attempt is retried once, and a
        task the model cannot do is dropped instead of retried forever.
        """
        ready: dict[str, TaskModel] = {}
        for task in self._tasks:
            if task.status != "failed":
                continue
            if self._attempts.get(task.id, 0) >= MAX_TASK_ATTEMPTS:
                continue
            ready[task.id] = task
        return list(ready.values())

    def _within_attempt_cap(self, plan: Plan, already: set[str]) -> list[TaskModel]:
        """Drop *plan* tasks that must not be handed to a worker again.

        Two exclusions, both of them the attempt cap doing its job: an id already
        queued for a retry this cycle (the retried object is the one that runs), and an
        id that has spent all ``MAX_TASK_ATTEMPTS`` attempts. Without the second, a
        planner that keeps re-issuing the same id would restart a task the model has
        already failed at, forever, in an unbounded run.
        """
        fresh: list[TaskModel] = []
        capped: list[str] = []
        for task in plan.tasks:
            if task.id in already:
                continue
            if self._attempts.get(task.id, 0) >= MAX_TASK_ATTEMPTS:
                capped.append(task.id)
                continue
            fresh.append(task)
        if capped:
            message = (
                f"Skipping task(s) {', '.join(capped)}: all "
                f"{MAX_TASK_ATTEMPTS} attempts have been used."
            )
            self.console.print(f"[yellow]{message}[/yellow]")
            self._emit("warning", message, cycle=self._cycles, capped=capped)
            self._add_feedback(message)
        return fresh

    def _status_of(self, task_id: str, batch: dict[str, TaskModel]) -> str | None:
        """Status of *task_id*: this cycle's task first, else the newest earlier one."""
        task = batch.get(task_id)
        if task is not None:
            return task.status
        for earlier in reversed(self._tasks):
            if earlier.id == task_id:
                return earlier.status
        return None

    def _completion_reason(self) -> str | None:
        """Why the run is finished successfully, or ``None`` to run another cycle."""
        if self._test_result is None or not self._test_result.passed:
            return None
        if self._planned_nothing:
            return "The planner has nothing left to do and the tests pass."
        if not self._cycle_tasks:
            return None
        failed = [task for task in self._cycle_tasks if task.status != "completed"]
        if not failed:
            return f"All {len(self._cycle_tasks)} task(s) completed and the tests pass."
        if not self._retry_queue():
            # Green suite plus nothing left to try: another cycle would only repeat the
            # same failures. Stop, but say plainly which tasks did not complete.
            stuck = ", ".join(task.id for task in failed)
            return (
                f"Tests pass but {len(failed)} task(s) never completed "
                f"({stuck}); they are past the {MAX_TASK_ATTEMPTS}-attempt cap, so "
                "another cycle could not change the verdict."
            )
        return None

    def _stop_reason(self) -> str | None:
        """Why the loop must stop without completion, or ``None`` to continue."""
        limit = self.settings.max_cycles
        if limit and self._cycles >= limit:
            return (
                "Reached the configured cycle limit "
                f"(settings.max_cycles={limit}) after {self._cycles} cycle(s)."
            )
        if self._empty_plan_streak >= MAX_EMPTY_PLAN_STREAK:
            return (
                "Stagnation: the planner returned no tasks "
                f"{self._empty_plan_streak} cycles in a row while the tests were still "
                "failing."
            )
        return None

    # ---------------------------------------------------------------- plumbing

    def _report_cycle_end(self, cycle: int) -> None:
        completed = sum(1 for task in self._cycle_tasks if task.status == "completed")
        failed = sum(1 for task in self._cycle_tasks if task.status == "failed")
        message = (
            f"Cycle {cycle} finished: {completed} completed, {failed} failed, "
            f"{len(self._cycle_tasks)} planned."
        )
        self.console.print(f"[dim]{message}[/dim]")
        self._emit("cycle_done", message, cycle=cycle, completed=completed, failed=failed)

    def _final_message(self, reason: str) -> str:
        if self._complete:
            return f"Goal complete after {self._cycles} cycle(s). {reason}"
        if self._cancelled:
            return f"Run cancelled after {self._cycles} cycle(s). {reason}"
        return f"Goal not complete after {self._cycles} cycle(s). {reason}"

    def _emit(self, kind: str, message: str, **fields: Any) -> None:
        """Hand one JSON-serialisable event to ``on_event``.

        The callback is consumer code (a websocket sink, a test) and must never be able
        to break the loop, so every emission is guarded: a raising sink loses one event,
        not the run.
        """
        if self._on_event is None:
            return
        event: dict[str, Any] = {"type": kind, "message": message}
        event.update(fields)
        try:
            self._on_event(event)
        except Exception:  # a broken sink must not stop the loop
            pass

    @contextmanager
    def _spinner(self, message: str) -> Iterator[None]:
        """Show *message* as a spinner on a terminal, as a plain line otherwise.

        rich allows only one active ``Live`` display per ``Console``, so two concurrent
        runs on the same console would raise ``LiveError``. Only a terminal gets the
        spinner.
        """
        if self.console.is_terminal:
            with Status(message, console=self.console):
                yield
            return
        self.console.print(message)
        yield

    async def _close_client(self) -> None:
        """Close the client only when this orchestrator created it."""
        if not self._owns_client:
            return
        try:
            await self.client.aclose()
        except Exception as exc:  # a failed close must not mask the run's outcome
            self._emit(
                "warning",
                f"Could not close the LLM client: {type(exc).__name__}: {exc}",
                error=f"{type(exc).__name__}: {exc}",
            )


def _dependency_cycle(tasks: Sequence[TaskModel]) -> list[str] | None:
    """Return one dependency cycle among *tasks* as a list of ids, or ``None``.

    :class:`~furrow.config.Plan` rejects duplicate and dangling ids but happily accepts
    a cycle, and a topological scheduler on top of that cycle would wait forever for a
    layer that can never become ready. This is the deadlock guard: it runs once per
    plan, before any task is launched.
    """
    by_id = {task.id: task for task in tasks}
    colour = dict.fromkeys(by_id, _UNRESOLVED)
    path: list[str] = []

    def deps_of(task_id: str) -> list[str]:
        return [dep for dep in by_id[task_id].dependencies if dep in by_id]

    for root in by_id:
        if colour[root] != _UNRESOLVED:
            continue
        colour[root] = _ACTIVE
        path.append(root)
        stack: list[tuple[str, list[str]]] = [(root, deps_of(root))]
        while stack:
            node, deps = stack[-1]
            for dep in deps:
                if colour[dep] == _ACTIVE:
                    return [*path[path.index(dep) :], dep]
                if colour[dep] == _UNRESOLVED:
                    colour[dep] = _ACTIVE
                    path.append(dep)
                    stack.append((dep, deps_of(dep)))
                    break
            else:
                colour[node] = _SETTLED
                path.pop()
                stack.pop()
    return None
