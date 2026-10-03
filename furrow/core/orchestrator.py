"""The Furrow loop: plan -> execute (dependency aware) -> test -> repeat.

This module owns the only state that survives a cycle. Every plan it produces is
kept in :attr:`Orchestrator.history`, completed tasks are tallied in
:attr:`Orchestrator.completed_total`, and the newest :class:`~furrow.config.TestResult`
is held in :attr:`Orchestrator.last_test_result`, so a cycle never starts from a
blank slate. A finished run reports how it ended as a :class:`RunOutcome`, so a
caller can tell a completed goal from a run that died (e.g. a planner outage).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

console = Console()

#: Anti-spin guard: stop after this many consecutive cycles that complete no *new* work.
#:
#: "New" is measured rather than assumed. Every completed task contributes a stable
#: signature (:func:`_task_signature`), and a cycle only counts as progress if it finished
#: a task whose signature this run had not completed before. Counting raw completions
#: cannot see the commonest way a loop gets stuck -- a planner that keeps re-issuing the
#: same work, which completes a task on every cycle and would spin forever -- whereas
#: every task the run has genuinely not done before still resets the counter.
MAX_NO_PROGRESS_CYCLES = 3

#: Consecutive planner failures tolerated before the run is abandoned.
MAX_PLANNER_FAILURES = 3

#: Progress hook used by streaming consumers (e.g. the websocket layer).
EventHook = Callable[[str, dict[str, Any]], None]

#: The halting paths the loop can take, in the order :meth:`Orchestrator._halt` tests them.
HALT_MAX_CYCLES = "max_cycles"
HALT_GOAL_COMPLETE = "goal_complete"
HALT_PLANNER_FAILED = "planner_failed"
HALT_NO_PROGRESS = "no_progress"


#: Stable identity of one unit of work: what the task says to do, and on which files.
_TaskSignature = tuple[str, tuple[str, ...]]


def _task_signature(task: TaskModel) -> _TaskSignature:
    """Return the signature that identifies ``task``'s work across the whole run.

    Two tasks are the same work when they ask for the same thing about the same files, so
    a planner that re-plans work the run already finished produces the same signature and
    is recognised as a repeat. The task id is deliberately excluded: a planner that renames
    or renumbers its tasks every cycle is still re-doing the same work.
    """
    return (task.description, tuple(sorted(set(task.files))))


def _unique(values: list[str]) -> list[str]:
    """De-duplicate ``values``, keeping first-seen order.

    A model may repeat a dependency id, and the same edge must only be reported once.
    """
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass(frozen=True)
class RunOutcome:
    """How a run ended. Returned by :meth:`Orchestrator.run`.

    Attributes:
        stop_reason: Human readable line saying why the loop halted.
        succeeded: Whether the goal was genuinely completed. ``False`` means the run
            failed, so a caller (CLI, CI job, scheduler) can act on it: the loop no
            longer "always exits cleanly", which used to hide total failures.
        cycles: Cycles that ran.
        completed: Tasks that completed over the whole run.
    """

    stop_reason: str
    succeeded: bool
    cycles: int
    completed: int


@dataclass(frozen=True)
class _Halt:
    """Internal stop decision: which path fired, its message, and the success verdict."""

    kind: str
    reason: str
    succeeded: bool


class Orchestrator:
    """Runs the infinite parallel development loop for a single goal.

    Args:
        goal: The original objective. Treated as immutable for the whole run.
        client: Shared LLM client; one is created when omitted.
        settings: Configuration override; the global settings object is used by default.
        on_event: Optional ``(event_name, payload)`` callback invoked at the natural
            points of every cycle: ``cycle_start``, ``plan``, ``task``, ``test`` and
            ``cycle_end``. Failures raised by the callback are swallowed.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        # The original objective is never rewritten; ``focus`` is the working objective.
        self.goal = goal
        self.focus = goal
        self.client = client or LLMClient()
        self.settings = settings if settings is not None else global_settings
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0

        # --- persistent state, survives across cycles -------------------------------
        self.history: list[Plan] = []
        self.tasks: list[TaskModel] = []
        self.completed_total = 0
        self.last_test_result: TestResult | None = None

        # --- bookkeeping for the stop condition -------------------------------------
        self._cycle_completed = 0
        self._cycle_progress = False
        self._no_progress_cycles = 0
        self._planner_failures = 0
        #: Signatures of every task completed so far, so a cycle can be judged on whether it
        #: did *new* work instead of merely on how many tasks it finished.
        self._completed_signatures: set[_TaskSignature] = set()

        # --- how the run ended; both filled in by run() when the loop exits --------
        self.stop_reason: str = ""
        self._outcome: RunOutcome | None = None

    # ------------------------------------------------------------------ main loop --

    @property
    def succeeded(self) -> bool:
        """Whether the finished run genuinely completed the goal.

        ``False`` until :meth:`run` exits, and ``False`` for a run that failed
        outright; see :meth:`_halt` for the rule.
        """
        return self._outcome is not None and self._outcome.succeeded

    @property
    def outcome(self) -> RunOutcome | None:
        """The :class:`RunOutcome` of the finished run, or ``None`` while it is running."""
        return self._outcome

    async def run(self) -> RunOutcome:
        """Drive the loop to a stop and report how it ended.

        Returns a :class:`RunOutcome`; callers that ignore the return value keep the
        previous behaviour.
        """
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._cycle_completed = 0
            self._cycle_progress = False
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle_start", {"cycle": self.cycles})

            await self._cycle()

            # A cycle only counts as progress if it completed a task this run had not
            # already completed. That strictly subsumes "completed nothing": a cycle that
            # finished nothing new is still a cycle that finished nothing new, and one that
            # repeats earlier work now trips the guard a count of completions would miss.
            if self._cycle_progress:
                self._no_progress_cycles = 0
            else:
                self._no_progress_cycles += 1
            self._emit(
                "cycle_end",
                {
                    "cycle": self.cycles,
                    "completed": self._cycle_completed,
                    "completed_total": self.completed_total,
                    "tests_passed": None if self.last_test_result is None else self.last_test_result.passed,
                },
            )

            halt = self._halt()
            if halt is not None:
                outcome = RunOutcome(
                    stop_reason=halt.reason,
                    succeeded=halt.succeeded,
                    cycles=self.cycles,
                    completed=self.completed_total,
                )
                self._outcome = outcome
                self.stop_reason = halt.reason
                colour = "bold green" if halt.succeeded else "bold red"
                console.print(f"[{colour}]{halt.reason}[/{colour}]")
                return outcome

    async def _cycle(self) -> None:
        plan = await self._plan()
        if plan is None:
            # Planning failed; _plan has already reported it and budgeted the retries.
            return

        self.history.append(plan)
        self.tasks.extend(plan.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            {
                "cycle": self.cycles,
                "plan": plan.model_dump(),
                "task_count": len(plan.tasks),
                "rationale": plan.rationale,
            },
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
        else:
            await self._execute_plan(plan)

        await self._test(plan)

    # -------------------------------------------------------------------- planning --

    async def _plan(self) -> Plan | None:
        """Ask the planner for work; return ``None`` when the planner failed.

        The planner raises ``ValueError`` on unparseable LLM output, which is a normal
        transient failure: it is reported and retried on the next cycle instead of
        taking the whole run down.
        """
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.focus)
        except Exception as exc:  # noqa: BLE001 - a bad LLM reply must not kill the run
            self._planner_failures += 1
            console.print(
                f"[red]Planning failed ({self._planner_failures}/{MAX_PLANNER_FAILURES}): {exc}[/red]"
            )
            self._emit("plan", {"cycle": self.cycles, "error": str(exc), "task_count": 0})
            return None

        self._planner_failures = 0
        return plan

    # ------------------------------------------------------------------- execution --

    async def _execute_plan(self, plan: Plan) -> None:
        """Run ``plan`` honouring ``TaskModel.dependencies``.

        Tasks are grouped into topological levels; a task only starts once every one of
        its dependencies has reached ``status == "completed"`` in this plan. Within a
        level at most ``max_parallel_tasks`` workers run concurrently.
        """
        # The graph is normalised once per plan so the scheduler and the readiness check
        # below cannot disagree about which edges count (see _normalised_graph).
        graph = self._normalised_graph(plan.tasks)
        levels = self._dependency_levels(plan.tasks, graph)
        by_id = {task.id: task for task in plan.tasks}

        for level in levels:
            runnable: list[TaskModel] = []
            for task in level:
                blocked_by = self._blocked_by(task, by_id, graph)
                if blocked_by is not None:
                    task.status = "skipped"
                    task.result = f"Skipped: dependency {blocked_by} did not complete."
                    console.print(f"[yellow]Task {task.id} skipped: {task.result}[/yellow]")
                    self._emit("task", self._task_event(task))
                    continue
                runnable.append(task)

            if not runnable:
                continue

            parallel_limit = max(1, int(getattr(self.settings, "max_parallel_tasks", 5)))
            semaphore = asyncio.Semaphore(parallel_limit)
            with Status(
                f"[bold yellow]Executing {len(runnable)} task(s) "
                f"(max {parallel_limit} in parallel)...",
                console=console,
            ):
                results = await asyncio.gather(
                    *(self._run_task(task, semaphore) for task in runnable),
                    return_exceptions=True,
                )

            for task, result in zip(runnable, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self.completed_total += 1
                    self._cycle_completed += 1
                    if self._record_completed(task):
                        self._cycle_progress = True
                    console.print(f"[green]Task {task.id} completed[/green]")
                self._emit("task", self._task_event(task))

    def _record_completed(self, task: TaskModel) -> bool:
        """Remember a completed task's signature; return whether it is new to this run.

        Args:
            task: The task that just completed.

        Returns:
            ``True`` when this signature had not been completed before in this run, i.e.
            the cycle did new work; ``False`` when the run had already done this work, so
            the task is a repeat however many tasks the cycle happened to finish.
        """
        signature = _task_signature(task)
        if signature in self._completed_signatures:
            return False
        self._completed_signatures.add(signature)
        return True

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _normalised_graph(self, tasks: list[TaskModel]) -> dict[str, list[str]]:
        """Normalise a plan's dependency edges once, per task id.

        The returned mapping is the single source of truth for the plan: every task id
        maps to the dependency ids that actually exist in the plan and are not the task
        itself. Two classes of unusable edge are reported and dropped, once each:

        * a self reference -- a task listing itself in its own ``dependencies``, and
        * a dangling edge -- a dependency id no task in the plan provides.

        Every other consumer of the plan's edges must read this mapping rather than the
        raw :attr:`TaskModel.dependencies`. Reading the raw list is what made a
        self-dependent task *scheduled* to run by :meth:`_dependency_levels` and then
        immediately *skipped* by :meth:`_blocked_by`, which judged the task blocked by
        its own ``pending`` status: one plan, two contradictory verdicts, and the task
        vanished without running.

        Args:
            tasks: The tasks of one plan.

        Returns:
            A mapping of task id to its usable dependency ids, in plan order.
        """
        by_id = {task.id for task in tasks}
        graph: dict[str, list[str]] = {}
        for task in tasks:
            kept: list[str] = []
            for dep in _unique(task.dependencies):
                if dep == task.id:
                    console.print(f"[yellow]Task {task.id} depends on itself; dropping edge.[/yellow]")
                    continue
                if dep not in by_id:
                    console.print(
                        f"[yellow]Task {task.id} depends on unknown task {dep!r}; dropping edge.[/yellow]"
                    )
                    continue
                kept.append(dep)
            graph[task.id] = kept
        return graph

    def _dependency_levels(
        self, tasks: list[TaskModel], graph: dict[str, list[str]] | None = None
    ) -> list[list[TaskModel]]:
        """Group ``tasks`` into topological waves (Kahn layering).

        Dependency cycles are reported and the tasks involved are run last, so a
        malformed plan still executes instead of hanging or raising. Self references and
        unknown dependency ids are dropped by :meth:`_normalised_graph`; pass its result
        as ``graph`` to build it once and warn once.

        Args:
            tasks: The tasks of one plan.
            graph: Normalised edges from :meth:`_normalised_graph`; computed from ``tasks``
                when omitted.
        """
        by_id = {task.id: task for task in tasks}
        if graph is None:
            graph = self._normalised_graph(tasks)
        pending: dict[str, set[str]] = {task_id: set(deps) for task_id, deps in graph.items()}
        dependents: dict[str, list[str]] = {}

        for task in tasks:
            for dep in graph.get(task.id, ()):
                dependents.setdefault(dep, []).append(task.id)

        levels: list[list[TaskModel]] = []
        frontier = [task.id for task in tasks if not pending.get(task.id)]
        while frontier:
            levels.append([by_id[task_id] for task_id in frontier])
            next_frontier: list[str] = []
            for task_id in frontier:
                for dependent_id in dependents.get(task_id, ()):
                    pending[dependent_id].discard(task_id)
                    if not pending[dependent_id]:
                        next_frontier.append(dependent_id)
            frontier = next_frontier

        stuck = [task for task in tasks if pending.get(task.id)]
        if stuck:
            ids = ", ".join(task.id for task in stuck)
            console.print(f"[yellow]Dependency cycle detected among: {ids}. Running them last.[/yellow]")
            levels.append(stuck)
        return levels

    def _blocked_by(
        self, task: TaskModel, by_id: dict[str, TaskModel], deps: dict[str, list[str]] | None = None
    ) -> str | None:
        """Return the first dependency of ``task`` that is not ``completed``, if any.

        Only edges that survived normalisation count, so a task is never blocked by
        itself or by an id the plan does not contain.

        Args:
            task: The task asking to run.
            by_id: The plan's tasks by id.
            deps: Normalised edges from :meth:`_normalised_graph`; derived from ``task``
                and ``by_id`` when omitted.
        """
        if deps is None:
            usable = [dep for dep in _unique(task.dependencies) if dep in by_id and dep != task.id]
        else:
            usable = deps.get(task.id, [])
        for dep in usable:
            dep_task = by_id[dep]
            if dep_task.status != "completed":
                return f"{dep} ({dep_task.status})"
        return None

    # --------------------------------------------------------------------- testing --

    async def _test(self, plan: Plan) -> None:
        """Run the tester, store the result and refine the working focus on failure."""
        # The tester always sees the original goal, plus the current focus.
        context = f"Original goal:\n{self.goal}\n\nCurrent focus:\n{self.focus}"
        try:
            with Status("[bold yellow]Testing...", console=console):
                test_result = await TesterAgent(client=self.client).run(context, plan.tasks)
        except Exception as exc:  # noqa: BLE001 - a broken test run must not kill the run
            console.print(f"[red]Testing failed: {exc}[/red]")
            self._emit("test", {"cycle": self.cycles, "error": str(exc), "passed": None})
            return

        self.last_test_result = test_result
        self._emit(
            "test",
            {"cycle": self.cycles, "result": test_result.model_dump(), "passed": test_result.passed},
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")
        console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        # Only the working focus is refined; ``self.goal`` keeps the original objective.
        details = "\n".join(test_result.failures) or test_result.summary
        self.focus = f"Original goal:\n{self.goal}\n\nFix failing tests:\n{details}"

    # ---------------------------------------------------------------- termination --

    def _halt(self) -> _Halt | None:
        """Return the stop decision, or ``None`` to keep looping.

        The loop is deliberately *not* terminated by a successful cycle: it runs until
        one of these conditions holds. ``max_cycles`` is the explicit user-facing kill
        switch (``0``, the default, means unlimited); the anti-spin guard is the backstop
        that keeps an unlimited run finite even when every cycle looks healthy.

        * ``max_cycles`` reached (``0``, the default, means unlimited),
        * the planner returns zero tasks while the tests pass: the goal is complete,
        * the planner keeps failing,
        * no *new* work completed for :data:`MAX_NO_PROGRESS_CYCLES` consecutive cycles
          (anti-spin): the run finished a task in none of them, or only re-finished work it
          had already finished, so a longer run would keep spending tokens to re-derive the
          same result.

        Success rule (a run is *not* a success just because it stopped cleanly):

        * the goal-complete path is a success by definition -- zero tasks were left
          and the tests pass;
        * for any other halt, the run succeeded only if it completed at least one task
          **and** the tests were passing, so running out of budget mid-way is a
          legitimate partial outcome rather than a failure;
        * a planner outage is never a success, even with an earlier green test result
          and completed tasks: the last cycles produced nothing at all;
        * the anti-spin guard with zero completed tasks is a failure (nothing was
          achieved), while the same guard after real work is a partial success.
        """
        max_cycles = int(getattr(self.settings, "max_cycles", 0) or 0)
        if max_cycles > 0 and self.cycles >= max_cycles:
            return _Halt(
                HALT_MAX_CYCLES,
                f"Reached max_cycles ({max_cycles}). Halting.",
                self._made_progress(),
            )

        if self.history and not self.history[-1].tasks and self._tests_passing():
            return _Halt(HALT_GOAL_COMPLETE, "Goal complete. Halting.", True)

        if self._planner_failures >= MAX_PLANNER_FAILURES:
            return _Halt(
                HALT_PLANNER_FAILED,
                f"Planner failed {self._planner_failures} cycles in a row. Halting.",
                False,
            )

        if self._no_progress_cycles >= MAX_NO_PROGRESS_CYCLES:
            return _Halt(
                HALT_NO_PROGRESS,
                f"No new work in {self._no_progress_cycles} consecutive cycles "
                "(the loop repeated already-completed work). Halting.",
                self._made_progress(),
            )

        return None

    def _made_progress(self) -> bool:
        """Whether the run accomplished anything worth reporting as success."""
        return self.completed_total > 0 and self._tests_passing()

    def _stop_reason(self) -> str | None:
        """Return why the loop should stop, or ``None`` to keep looping.

        Message-only view of :meth:`_halt`, kept for callers that only need the text.
        """
        halt = self._halt()
        return None if halt is None else halt.reason

    # ------------------------------------------------------------------- accessors --

    def _tests_passing(self) -> bool:
        return self.last_test_result is not None and self.last_test_result.passed

    # ---------------------------------------------------------------------- events --

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        """Invoke the progress hook; a raising callback must never break the loop."""
        if self.on_event is None:
            return
        try:
            self.on_event(event, payload)
        except Exception as exc:  # noqa: BLE001 - a raising hook must not break the loop
            console.print(f"[yellow]on_event hook failed during {event!r}: {exc}[/yellow]")

    def _task_event(self, task: TaskModel) -> dict[str, Any]:
        return {
            "cycle": self.cycles,
            "id": task.id,
            "status": task.status,
            "description": task.description,
            "result": task.result,
        }