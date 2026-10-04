"""The infinite parallel development loop that drives Furrow.

:class:`Orchestrator` owns one run: it repeatedly asks a planner for work,
executes the resulting tasks with dependency-aware bounded parallelism, runs the
test suite, and folds any failure back into the next planning round. State lives
on the instance, so progress survives from one cycle to the next.

Three properties make the loop safe to run unattended and to drive from outside
a terminal:

* **Bounded.** ``settings.max_cycles`` caps the number of cycles (``0`` means
  unlimited) and two stall guards halt a self-repair loop that keeps producing
  failures without finishing anything new or without fixing a red test suite.
* **Observable.** Every lifecycle step is published as a
    :class:`~furrow.core.events.FurrowEvent` through the optional ``on_event``
    callback, and terminal progress is printed with :mod:`rich`. Both are
    best-effort: a misbehaving listener or a non-terminal console can never
    break the loop.
* **Resilient.** A planner that returns malformed JSON, a provider that raises
    :class:`~furrow.llm.LLMError`, or any other unexpected exception is recorded
    as an error and the loop continues instead of aborting the run.

Example:
    ```python
    events = []
    orchestrator = Orchestrator(goal="Add a health check endpoint", on_event=events.append)
    result = await orchestrator.run()
    if not result.completed:
        print(result.stopped_reason)
    ```
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus, get_settings
from furrow.core.events import EventType, FurrowEvent, RunResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import TestResult

console = Console()

_T = TypeVar("_T")


class _TaskOutcome(NamedTuple):
    """What one worker produced for a single task.

    Attributes:
        task: The registry entry that was executed.
        result: The worker's report, or ``None`` when it raised.
        error: The exception raised by the worker, or ``None`` on success.
        files_written: Paths the worker reported writing, best effort.
        errors: Non-fatal problems the worker reported, best effort.
    """

    task: TaskModel
    result: str | None
    error: BaseException | None
    files_written: list[str]
    errors: list[str]


class Orchestrator:
    """Run plan/execute/test cycles until a goal is reached or the run halts.

    Attributes:
        original_goal: The goal exactly as supplied by the caller. It is never
            modified, so a repair round cannot erase the real objective.
        current_goal: The goal handed to the planner for the next cycle. It is
            :attr:`original_goal` plus a repair preamble after a failing test
            run.
        tasks: Registry of every task produced in any cycle of this run. Tasks
            are reused across cycles by id, so their status, result and
            dependency information persist between rounds.
        settings: Configuration used for cycle limits, parallelism and agents.
        client: The shared :class:`~furrow.llm.LLMClient`, released by
            :meth:`run`.
        planner: The planner asked for each cycle's tasks.
        cycles: Number of cycles started so far.

    A run halts when the goal is reached, when ``settings.max_cycles`` is
    exhausted, when :attr:`MAX_UNPRODUCTIVE_CYCLES` consecutive cycles fail to
    finish a task that had never finished before, when
    :attr:`MAX_REPAIR_ATTEMPTS` consecutive repair cycles fail to fix a failing
    suite, or when :meth:`run` is cancelled.
    """

    MAX_UNPRODUCTIVE_CYCLES: int = 3
    MAX_REPAIR_ATTEMPTS: int = 3

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[FurrowEvent], None] | None = None,
    ) -> None:
        """Create an orchestrator for `goal`.

        Args:
            goal: The objective handed to the planner on every cycle.
            client: LLM client shared by the planner, workers and tester. When
                omitted a new one is built from ``settings``.
            settings: Runtime configuration. Defaults to the shared
                :func:`furrow.config.get_settings` singleton.
            on_event: Optional synchronous listener invoked once per lifecycle
                event. It is called from async context, so it must not block,
                and every invocation is guarded: an exception raised by the
                listener is swallowed rather than breaking the loop.
        """
        self.original_goal = goal
        self.current_goal = goal
        self.settings = settings or get_settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self._on_event = on_event
        self._status: Status | None = None
        self._repair_mode = False
        self._last_test_passed = False
        self._settled_task_ids: set[str] = set()
        self._unproductive_cycles = 0
        self._repair_attempts = 0
        self._cycles_failed = 0
        self._last_error: str | None = None

    @property
    def goal(self) -> str:
        """Return the goal the planner will use next.

        Returns:
            :attr:`current_goal`, which differs from :attr:`original_goal` while
            a repair preamble is in effect.
        """
        return self.current_goal

    @goal.setter
    def goal(self, value: str) -> None:
        """Replace the goal used for subsequent planning.

        Args:
            value: New planning goal. It does not change :attr:`original_goal`.
        """
        self.current_goal = value

    async def run(self) -> RunResult:
        """Run cycles until the goal is reached or the loop halts.

        The loop is bounded by ``settings.max_cycles`` (``0`` means unlimited)
        and by two stall guards: :attr:`MAX_UNPRODUCTIVE_CYCLES` consecutive
        cycles that finish nothing new, and :attr:`MAX_REPAIR_ATTEMPTS`
        consecutive repair cycles that leave the suite failing. A permanently
        failing loop therefore halts instead of spending money forever.
        Exceptions raised inside a cycle, including planner ``ValueError`` and
        :class:`~furrow.llm.LLMError`, are recorded and the loop continues; only
        cancellation escapes.

        Returns:
            A :class:`~furrow.core.events.RunResult` describing the run, with
            ``stopped_reason`` always populated.

        Raises:
            asyncio.CancelledError: If the surrounding task is cancelled, after
                the LLM client has been released.
        """
        self._emit(EventType.RUN_START, message=f"Goal: {self.original_goal}")
        self._echo(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        completed = False
        reason = "Run did not start."
        try:
            while True:
                if self._last_test_passed and (self._is_done() or not self._get_tasks()):
                    completed = True
                    reason = self._completion_reason()
                    break
                limit = self.settings.max_cycles
                if limit > 0 and self.cycles >= limit:
                    reason = f"Reached the configured max_cycles limit of {limit}."
                    break
                if self._unproductive_cycles >= self.MAX_UNPRODUCTIVE_CYCLES:
                    reason = (
                        f"Stopped after {self._unproductive_cycles} consecutive cycles "
                        "completed no new tasks (repair loop stalled)."
                    )
                    break
                if self._repair_attempts >= self.MAX_REPAIR_ATTEMPTS:
                    reason = (
                        f"Stopped after {self._repair_attempts} consecutive repair cycles "
                        "failed to fix the failing test suite."
                    )
                    break
                self.cycles += 1
                await self._run_one_cycle()
        except asyncio.CancelledError:
            self._emit(
                EventType.RUN_END, message="Run cancelled.", cancelled=True, cycles=self.cycles
            )
            raise
        except Exception as exc:
            self._record_error(exc, phase="run")
            reason = f"Run aborted by an unexpected error: {self._last_error}"
        finally:
            self._release_client()

        result = RunResult(
            cycles=self.cycles,
            completed=completed,
            tasks_completed=self._count(TaskStatus.COMPLETED),
            tasks_failed=self._count(TaskStatus.FAILED),
            cycles_failed=self._cycles_failed,
            last_error=self._last_error,
            stopped_reason=reason,
        )
        self._echo(
            Panel.fit(
                f"{result.stopped_reason}\n"
                f"cycles={result.cycles} completed={result.tasks_completed} "
                f"failed={result.tasks_failed}",
                title="Run finished",
                border_style="green" if completed else "yellow",
            )
        )
        self._emit(EventType.RUN_END, message=reason, **result.model_dump(mode="json"))
        return result

    async def _cycle(self) -> None:
        """Plan, execute and test once, publishing every step as an event.

        Planned tasks are merged into :attr:`tasks` and executed with
        dependency-aware bounded parallelism, then the whole registry is tested.
        A failing test run switches on repair mode, which re-opens completed
        tasks on the next cycle rather than discarding their results, and
        prefixes :attr:`current_goal` with the failures while leaving
        :attr:`original_goal` intact. Repair cycles that leave the suite failing
        are counted against :attr:`MAX_REPAIR_ATTEMPTS`; the first measurement
        of a failing suite is not counted, because it is not yet an attempt to
        repair anything.

        Returns:
            None.

        Raises:
            ValueError: If the planner cannot parse the model response.
            LLMError: If an agent's LLM request fails after its retries.
        """
        self._emit(EventType.CYCLE_START, message=f"Cycle {self.cycles}")
        self._echo(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
        was_repairing = self._repair_mode

        plan = await self._with_status("Planning...", self.planner.plan(self.current_goal))
        self._echo(
            Panel(
                Pretty(plan.model_dump(mode="json")),
                title=f"Plan for cycle {self.cycles}",
                border_style="blue",
            )
        )
        self._emit(
            EventType.PLAN,
            message=plan.rationale,
            task_count=len(plan.tasks),
            tasks=[task.id for task in plan.tasks],
        )

        runnable = self._merge_plan(plan)
        if not runnable:
            self._echo("[yellow]No tasks to execute this cycle.[/yellow]")
        await self._with_status("Executing tasks...", self._execute(runnable))

        test_result = await self._with_status("Testing...", self._test(self.current_goal))
        if test_result is None:
            return
        if test_result.passed:
            self._last_test_passed = True
            self._repair_mode = False
            self._repair_attempts = 0
            self.current_goal = self.original_goal
            self._echo(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._last_test_passed = False
            self._repair_mode = True
            if was_repairing:
                self._repair_attempts += 1
            self._echo(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._echo(f"  • {failure}")
            self._set_repair_goal(test_result)
            self._echo("[yellow]Will attempt a repair in the next cycle.[/yellow]")
        self._emit(
            EventType.CYCLE_END,
            message=test_result.summary,
            tests_passed=test_result.passed,
            tasks_total=len(self._get_tasks()),
            tasks_completed=self._count(TaskStatus.COMPLETED),
            repair_attempts=self._repair_attempts,
        )

    def _get_tasks(self) -> list[TaskModel]:
        """Return the task registry accumulated across every cycle.

        Returns:
            The live :attr:`tasks` list, in the order tasks were first planned.
            Callers must treat the entries as mutable registry records.
        """
        return self.tasks

    def _is_done(self) -> bool:
        """Return whether every registered task has completed.

        An empty registry is *not* done: nothing has been accomplished yet, so
        treating it as finished would let a planner that returns no tasks end the
        run immediately. :meth:`run` additionally requires the most recent test
        run to pass before it reports ``RunResult.completed``.

        Returns:
            True when the registry is non-empty and every task's status is
            ``COMPLETED``. Any ``FAILED``, ``SKIPPED``, ``RUNNING`` or
            ``PENDING`` task, including one blocked by a failed dependency,
            makes this False so a repair cycle gets a chance.
        """
        tasks = self._get_tasks()
        if not tasks:
            return False
        return all(task.status is TaskStatus.COMPLETED for task in tasks)

    async def _run_one_cycle(self) -> None:
        """Run one cycle, absorbing any failure so the loop keeps its bounds.

        Records the cycle's error, marks the run as untested so a failed cycle
        can never be reported as a completed goal, and updates the consecutive
        unproductive-cycle counter used by the stall guard.

        Returns:
            None. Failures are reported through events and :attr:`_last_error`
            instead of propagating.
        """
        settled_before = len(self._settled_task_ids)
        try:
            await self._cycle()
        except Exception as exc:
            self._cycles_failed += 1
            self._last_test_passed = False
            self._record_error(exc, phase="cycle")
        if len(self._settled_task_ids) > settled_before:
            self._unproductive_cycles = 0
        else:
            self._unproductive_cycles += 1

    async def _with_status(self, message: str, work: Awaitable[_T]) -> _T:
        """Await `work` under a live spinner that can be advanced by callees.

        Args:
            message: Initial spinner text.
            work: The awaitable to run while the spinner is live. Nested work
                may call :meth:`_set_status` to report finer-grained progress.

        Returns:
            Whatever `work` returns.
        """
        with Status(f"[bold yellow]{message}[/bold yellow]", console=console) as status:
            self._status = status
            try:
                return await work
            finally:
                self._status = None

    def _merge_plan(self, plan: Plan) -> list[TaskModel]:
        """Fold a plan into the registry and select the tasks to run now.

        Task identity is the planner-supplied id: a re-planned task reuses its
        registry entry, so status, results and dependencies persist across
        cycles. Tasks that already completed are left untouched unless repair
        mode is active, in which case they are re-opened because a failing test
        run means their work is not sufficient after all.

        Args:
            plan: The plan produced for this cycle.

        Returns:
            The tasks to execute in this cycle, without duplicates.
        """
        known = {task.id: task for task in self.tasks}
        runnable: list[TaskModel] = []
        for planned in plan.tasks:
            existing = known.get(planned.id)
            if existing is None:
                existing = planned
                self.tasks.append(existing)
                known[planned.id] = existing
            else:
                existing.description = planned.description
                existing.files = list(planned.files)
                existing.dependencies = list(planned.dependencies)
            if existing.status is TaskStatus.COMPLETED and not self._repair_mode:
                continue
            if existing in runnable:
                continue
            existing.status = TaskStatus.PENDING
            runnable.append(existing)
        return runnable

    async def _execute(self, runnable: list[TaskModel]) -> None:
        """Run `runnable` in dependency-ordered waves of bounded concurrency.

        Independent tasks run together under an :class:`asyncio.Semaphore` sized
        by ``settings.max_parallel_tasks``; a task whose dependencies are not yet
        ``COMPLETED`` waits for the next wave. The scheduler cannot deadlock: a
        task whose dependency failed is ``SKIPPED`` with a message naming it, and
        when a wave produces nothing runnable the remaining tasks are ``SKIPPED``
        as unresolvable instead of waiting forever.
        :func:`asyncio.gather` is used with ``return_exceptions=True`` so one
        worker's failure never cancels its siblings.

        Args:
            runnable: Tasks selected by :meth:`_merge_plan` for this cycle.

        Returns:
            None. Every task ends this method in a terminal status.
        """
        registry = {task.id: task for task in self.tasks}
        remaining = list(runnable)
        pass_limit = len(runnable) + 8
        passes = 0
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        while remaining:
            passes += 1
            if passes > pass_limit:
                for task in remaining:
                    self._mark_skipped(task, "Scheduler gave up on unresolved dependencies")
                return
            ready, blocked, skipped = self._partition(remaining, registry)
            for task, reason in skipped:
                self._mark_skipped(task, reason)
            if not ready:
                for task in blocked:
                    self._mark_skipped(task, "Unresolved dependencies (dependency cycle?)")
                return
            self._set_status(
                f"Executing {len(ready)} of {len(self.tasks)} tasks "
                f"(max {self.settings.max_parallel_tasks} in parallel)..."
            )
            outcomes = await asyncio.gather(
                *(self._run_task(task, semaphore) for task in ready),
                return_exceptions=True,
            )
            for task, outcome in zip(ready, outcomes, strict=True):
                if isinstance(outcome, _TaskOutcome):
                    self._record_outcome(outcome)
                else:
                    self._mark_failed(task, f"{type(outcome).__name__}: {outcome}")
            remaining = blocked

    def _partition(
        self,
        remaining: list[TaskModel],
        registry: dict[str, TaskModel],
    ) -> tuple[list[TaskModel], list[TaskModel], list[tuple[TaskModel, str]]]:
        """Split `remaining` into runnable, waiting and skipped tasks.

        Args:
            remaining: Tasks still to be scheduled in this cycle.
            registry: Every known task by id, so dependencies satisfied in an
                earlier cycle are recognised.

        Returns:
            A ``(ready, blocked, skipped)`` triple. ``ready`` tasks have all
            their dependencies ``COMPLETED``, or have dependencies the registry
            does not know about; ``blocked`` tasks still wait on a scheduled
            dependency; ``skipped`` pairs name a task and the dependency that
            failed it.
        """
        ready: list[TaskModel] = []
        blocked: list[TaskModel] = []
        skipped: list[tuple[TaskModel, str]] = []
        for task in remaining:
            unmet: list[str] = []
            failed_dependency: str | None = None
            unknown_dependency: str | None = None
            for dependency_id in task.dependencies:
                dependency = registry.get(dependency_id)
                if dependency is None:
                    unknown_dependency = unknown_dependency or dependency_id
                    continue
                if dependency.status in {TaskStatus.FAILED, TaskStatus.SKIPPED}:
                    failed_dependency = failed_dependency or dependency_id
                elif dependency.status is not TaskStatus.COMPLETED:
                    unmet.append(dependency_id)
            if failed_dependency is not None:
                skipped.append((task, f"Dependency {failed_dependency} did not complete"))
                continue
            if unmet:
                blocked.append(task)
                continue
            if unknown_dependency is not None:
                self._emit(
                    EventType.ERROR,
                    message=(
                        f"Task {task.id} depends on unknown task {unknown_dependency!r}; "
                        "treating it as ready."
                    ),
                    task_id=task.id,
                    unknown_dependency=unknown_dependency,
                )
            ready.append(task)
        return ready, blocked, skipped

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> _TaskOutcome:
        """Execute one task behind `semaphore`, reporting rather than raising.

        Args:
            task: The registry entry to execute.
            semaphore: Bounds how many workers run concurrently.

        Returns:
            An :class:`_TaskOutcome` describing success, or the exception that
            the worker raised.
        """
        task.status = TaskStatus.RUNNING
        self._emit(
            EventType.TASK_START,
            message=task.description,
            task_id=task.id,
            files=task.files,
            dependencies=task.dependencies,
        )
        async with semaphore:
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            try:
                result = await worker.run()
            except Exception as exc:
                return self._outcome(task, None, exc, worker)
            return self._outcome(task, result, None, worker)

    def _outcome(
        self,
        task: TaskModel,
        result: str | None,
        error: BaseException | None,
        worker: Any,
    ) -> _TaskOutcome:
        """Bundle a worker result with whatever it reported about its work.

        Args:
            task: The task the worker was given.
            result: The worker's report, or ``None`` when it raised.
            error: The exception raised by the worker, or ``None``.
            worker: The worker instance, read for ``files_written`` and
                ``errors``. Both are read defensively so an agent that does not
                expose them cannot fail the task it just finished.

        Returns:
            The :class:`_TaskOutcome` to record.
        """
        files_written = list(getattr(worker, "files_written", None) or [])
        errors = list(getattr(worker, "errors", None) or [])
        return _TaskOutcome(
            task=task,
            result=result,
            error=error,
            files_written=files_written,
            errors=errors,
        )

    def _record_outcome(self, outcome: _TaskOutcome) -> None:
        """Persist a finished task and publish its terminal event.

        A task that completes for the first time joins the settled set, which is
        what the stall guard uses to tell real progress from a repair loop that
        merely redoes the same work.

        Args:
            outcome: The result of running one task.

        Returns:
            None.
        """
        task = outcome.task
        if outcome.error is not None:
            self._mark_failed(
                task,
                f"{type(outcome.error).__name__}: {outcome.error}",
                files_written=outcome.files_written,
                errors=outcome.errors,
            )
            return
        task.result = outcome.result
        task.status = TaskStatus.COMPLETED
        self._settled_task_ids.add(task.id)
        self._emit(
            EventType.TASK_COMPLETE,
            message=task.description,
            task_id=task.id,
            description=task.description,
            result=outcome.result,
            files_written=outcome.files_written,
            errors=outcome.errors,
        )
        self._echo(f"[green]Task {task.id} completed[/green]")

    def _mark_failed(self, task: TaskModel, message: str, **data: Any) -> None:
        """Mark a task ``FAILED`` and publish :attr:`EventType.TASK_FAILED`.

        Args:
            task: The task to fail.
            message: Why it failed; also stored as the task result.
            **data: Extra payload merged into the event's ``data``.

        Returns:
            None.
        """
        task.status = TaskStatus.FAILED
        task.result = message
        self._emit(EventType.TASK_FAILED, message=message, task_id=task.id, **data)
        self._echo(f"[red]Task {task.id} failed: {message}[/red]")

    def _mark_skipped(self, task: TaskModel, message: str) -> None:
        """Mark a task ``SKIPPED`` and publish :attr:`EventType.TASK_SKIPPED`.

        Args:
            task: The task to skip.
            message: Why it cannot run; also stored as the task result.

        Returns:
            None.
        """
        task.status = TaskStatus.SKIPPED
        task.result = message
        self._emit(
            EventType.TASK_SKIPPED,
            message=message,
            task_id=task.id,
            description=task.description,
        )
        self._echo(f"[yellow]Task {task.id} skipped: {message}[/yellow]")

    async def _test(self, goal: str) -> TestResult | None:
        """Run the test suite against the current registry state.

        Args:
            goal: The goal to report to the tester.

        Returns:
            The :class:`~furrow.config.TestResult` from the tester, or ``None``
            if the tester could not be constructed.
        """
        self._emit(EventType.TEST_START, message="Running the test suite.")
        try:
            tester: Any = TesterAgent(client=self.client, settings=self.settings)
        except Exception as exc:
            self._record_error(exc, phase="tester")
            return None
        test_result = await tester.run(goal, list(self._get_tasks()))
        self._emit(
            EventType.TEST_RESULT,
            message=test_result.summary,
            passed=test_result.passed,
            failures=list(test_result.failures),
        )
        return test_result

    def _set_repair_goal(self, test_result: TestResult) -> None:
        """Prefix :attr:`current_goal` with the failures to repair.

        :attr:`original_goal` is left untouched, so the repair instruction never
        displaces the real objective and later cycles still see it.

        Args:
            test_result: The failing test result to describe.

        Returns:
            None.
        """
        failures = "\n".join(f"- {failure}" for failure in test_result.failures)
        summary = test_result.summary or "the suite is failing"
        self.current_goal = (
            f"{self.original_goal}\n\n"
            f"The test suite is failing ({summary}). Repair the failures below "
            f"without losing sight of the goal above:\n{failures}"
        )

    def _completion_reason(self) -> str:
        """Return the reason recorded when the run reaches its goal.

        Returns:
            A sentence explaining why the loop stopped early.
        """
        if not self._get_tasks():
            return "Planner produced no remaining work and the test suite passed."
        total = len(self._get_tasks())
        return f"All {total} planned tasks completed and the test suite passed."

    def _record_error(self, exc: BaseException, phase: str) -> None:
        """Record an exception as an error event and in :attr:`_last_error`.

        Args:
            exc: The exception that was caught.
            phase: Where it happened, for example ``"cycle"`` or ``"run"``.

        Returns:
            None.
        """
        message = f"{type(exc).__name__}: {exc}"
        self._last_error = message
        self._emit(
            EventType.ERROR,
            message=f"{phase} failed: {message}",
            error_type=type(exc).__name__,
            phase=phase,
            cycle_number=self.cycles,
        )
        self._echo(f"[red]{phase} failed: {message}[/red]")

    def _emit(
        self,
        event_type: EventType,
        message: str = "",
        task_id: str | None = None,
        **data: Any,
    ) -> None:
        """Publish one lifecycle event to the ``on_event`` listener.

        Payload contract: **the payload is flat**. ``FurrowEvent.data`` is a
        single flat mapping, so a caller's keys become its keys directly, next to
        its siblings (``self._emit(EventType.TASK_FAILED, message=m, task_id=t,
        error_type="Boom")``). Nothing may nest a payload under a ``data`` key:
        ``FurrowEvent.data == {"data": {...}}`` is a wire format no consumer —
        including the browser UI, which forwards this dict verbatim — can read,
        so it is always a bug at the call site rather than a shape to support.
        That mistake has shipped once already, hence the guard below: splatting
        (``**payload``) is the only way to pass a mapping, and a literal
        ``data=`` keyword now fails loudly here instead of silently emitting a
        malformed frame nobody notices until a client mis-renders.

        Args:
            event_type: Which lifecycle step occurred.
            message: Human-readable summary of the step.
            task_id: Task the step concerns, when applicable.
            **data: Extra payload keys flattened into the event's ``data``.

        Returns:
            None. A listener that raises is ignored so it cannot break the run.

        Raises:
            TypeError: If ``data`` is passed as a keyword rather than splatted.
        """
        if "data" in data:
            raise TypeError(
                "_emit() does not accept a 'data' keyword; splat the payload instead "
                "(e.g. **payload) so it stays flat beside its siblings."
            )
        if self._on_event is None:
            return
        event = FurrowEvent(
            type=event_type,
            message=message,
            cycle=self.cycles,
            task_id=task_id,
            data=data,
        )
        try:
            self._on_event(event)
        except Exception:
            return

    def _set_status(self, message: str) -> None:
        """Advance the live spinner, if a cycle currently owns one.

        Args:
            message: New spinner text.

        Returns:
            None.
        """
        status = self._status
        if status is None:
            return
        try:
            status.update(f"[bold yellow]{message}[/bold yellow]")
        except Exception:
            self._status = None

    def _echo(self, renderable: RenderableType) -> None:
        """Print `renderable` without letting the console break the run.

        Args:
            renderable: Anything :mod:`rich` can render.

        Returns:
            None.
        """
        try:
            console.print(renderable)
        except Exception:
            return

    def _release_client(self) -> None:
        """Release the LLM client so SDK connections are not leaked.

        Returns:
            None. A failing ``release`` is reported as an error event rather
            than raised, because it must not mask the run result.
        """
        release = getattr(self.client, "release", None)
        if release is None:
            return
        try:
            release()
        except Exception as exc:
            self._emit(
                EventType.ERROR,
                message=f"Releasing the LLM client failed: {type(exc).__name__}: {exc}",
                phase="release",
            )

    def _count(self, status: TaskStatus) -> int:
        """Return how many registered tasks are in `status`.

        Args:
            status: The status to count.

        Returns:
            The number of registry entries whose status matches.
        """
        return sum(1 for task in self._get_tasks() if task.status is status)
