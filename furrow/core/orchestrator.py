"""The furrow orchestration loop: plan, execute, test, repeat - until the goal holds.

The loop is deliberately *bounded* even though the product promise is an "infinite
parallel development loop".  Around an LLM planner an unbounded loop is not a
feature, it is a hot loop that spends money, so every exit is explicit, states its
reason, and is published on the event stream as a ``done`` event.

Three things this module exists to guarantee, all of which the previous version got
wrong:

* **State is real and persistent.**  ``_get_tasks()`` returned a literal ``[]``, so
  ``_is_done()`` evaluated ``0 >= 0`` and ``run()`` printed "Goal complete" after
  exactly one cycle - the ``while True:`` was dead code.  ``self.tasks`` is now the
  authoritative list and a new plan is merged into it, not thrown over it.
* **Halting means the goal is proven.**  ``test_result.passed`` used to be computed
  and discarded, so a red test suite could not keep the loop alive.  Every task
  COMPLETED *and* the latest test run green is the only "done".
* **The goal is never lost.**  A failing cycle used to overwrite the goal with
  ``"Fix failing tests:\\n" + failures``, which on the JSON-parse-failure path became
  the content-free literal ``"Fix failing tests:\\n"``.  The working goal is always
  rebuilt from :attr:`Orchestrator.original_goal`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Final, TypeAlias

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TaskStatus, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

#: ``done.reason`` values.  Stable strings: a WebSocket client keys off them, so do
#: not reword these without updating the contract.
HALT_GOAL_COMPLETE: Final[str] = "goal complete"
HALT_MAX_CYCLES: Final[str] = "max_cycles reached"
HALT_NO_PROGRESS: Final[str] = "no progress"
HALT_NO_TASKS: Final[str] = "no tasks planned"
HALT_ERROR: Final[str] = "error"

#: Consecutive cycles that may achieve nothing at all before the loop gives
#: up: the halt fires *on* the STALL_LIMIT-th dead cycle in a row.  The first
#: cycle always counts as progress (it sets the baseline from an empty state),
#: so a fully stalled unlimited run halts at STALL_LIMIT + 1 == 4 cycles: one
#: baseline cycle, then three dead ones.
#: ``Settings.max_cycles == 0`` means unlimited, and unlimited plus a planner
#: that keeps re-planning the same failing work is an infinite hot loop; three
#: dead cycles in a row is enough to prove the loop has nothing left to offer.
STALL_LIMIT: Final[int] = 3

#: Fingerprint of everything a cycle can change, used to detect a stalled loop:
#: ``(completed, failed, test outcome)``.  Worker prose is deliberately excluded - it
#: changes on every cycle even when nothing was actually achieved.
StateToken: TypeAlias = tuple[int, int, tuple[bool, tuple[str, ...]] | None]


class Orchestrator:
    """Runs plan -> execute -> test cycles until the goal is proven satisfied.

    Args:
        goal: The objective.  Kept verbatim as :attr:`original_goal` and never
            overwritten, so any later cycle can restate what the run is for.
        client: Shared LLM client.  Every agent receives this same instance, so one
            set of provider connections is pooled instead of one pool per task.
        settings: Configuration.  Defaults to the injected client's own settings,
            then to the module singleton.
        console: Rich console for human-facing output.  Inject ``Console(file=...)``
            to capture output in tests, or a quiet one when the web tier must not
            touch stdout.
        on_event: Optional sink for the JSON-serialisable event stream, whose
            history is always kept in :attr:`events`.  A sink that raises can never
            end the run.
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
        self.settings: Settings = self._resolve_settings(client, settings)
        self.client: LLMClient = client if client is not None else LLMClient(
            settings=self.settings
        )
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.original_goal: str = goal
        # The *working* goal: restated from original_goal plus the latest failures,
        # never instead of the original objective.
        self.goal: str = goal
        #: The authoritative task list.  ``_get_tasks()`` is just a view of this.
        self.tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None
        self.cycles: int = 0
        #: Every event emitted by this run, oldest first.
        self.events: list[dict[str, Any]] = []
        self.console: Console = console if console is not None else Console()
        self.on_event = on_event
        self._stalled_cycles: int = 0
        self._last_state: StateToken | None = None

    @staticmethod
    def _resolve_settings(client: LLMClient | None, settings: Settings | None) -> Settings:
        """Pick the effective settings: explicit, else the client's, else the global.

        Matches the precedence the agents already use for their own settings, so the
        orchestrator's view of ``max_cycles``/``max_parallel_tasks`` is the view the
        agents act on.
        """
        if settings is not None:
            return settings
        client_settings = getattr(client, "settings", None)
        if isinstance(client_settings, Settings):
            return client_settings
        return global_settings

    # ------------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        """Run cycles until the goal is proven, a cap is hit, or the loop stalls.

        Never propagates an agent failure: every exit emits a ``done`` event naming
        the reason, so a caller (CLI, web socket) always learns how the run ended.
        """
        self._print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"
            )
        )
        max_cycles = self.settings.max_cycles
        reason = HALT_MAX_CYCLES  # 0 means unlimited, so this is only a fallback
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._print(f"[yellow]Cycle cap ({max_cycles}) reached.[/yellow]")
                reason = HALT_MAX_CYCLES
                break
            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                halt = await self._cycle()
            except Exception as exc:  # a bug here must not eat the whole process
                self._error(f"Cycle {self.cycles} failed: {exc!r}")
                reason = HALT_ERROR
                break
            if halt is not None:
                reason = halt
                break
            if self._is_done():
                reason = HALT_GOAL_COMPLETE
                break
            if self._register_cycle_state():
                self._print(
                    f"[yellow]No progress for {self._stalled_cycles} consecutive cycle(s).[/yellow]"
                )
                reason = HALT_NO_PROGRESS
                break
        self._finish(reason)

    async def aclose(self) -> None:
        """Release the shared LLM client's connections.

        ``getattr`` rather than a direct call so an injected test double without
        ``aclose`` is not a crash.  :meth:`LLMClient.aclose` is idempotent, so the
        CLI and the web tier can call this as often as they like.
        """
        closer = getattr(self.client, "aclose", None)
        if closer is None:
            return
        await closer()

    # ---------------------------------------------------------------------- cycle

    async def _cycle(self) -> str | None:
        """Run one plan/execute/test cycle.

        Returns:
            A halt reason when this cycle must end the run, ``None`` to keep going.
        """
        self._emit({"type": "cycle_start", "cycle": self.cycles, "goal": self.goal})
        try:
            with Status("[bold yellow]Planning...", console=self.console):
                plan = await self.planner.plan(self.goal)
        except Exception as exc:
            # A planner/LLM error used to escape as a raw traceback and take the run
            # with it; there is nothing to retry against, so halt cleanly instead.
            self._error(f"Planning failed: {exc}")
            return HALT_ERROR

        self._emit(
            {
                "type": "plan",
                "tasks": [
                    {
                        "id": task.id,
                        "description": task.description,
                        "files": list(task.files),
                        "dependencies": list(task.dependencies),
                    }
                    for task in plan.tasks
                ],
                "rationale": plan.rationale,
            }
        )
        self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        if not plan.tasks:
            # Zero tasks is a dead planner, not progress: halt rather than spin.
            self._print("[yellow]Planner returned no tasks; nothing left to do.[/yellow]")
            return HALT_NO_TASKS

        self._merge_tasks(plan.tasks)
        await self._execute_tasks(self.tasks)
        test_result = await self._test(self.goal, self.tasks)
        self.test_result = test_result
        self._emit(
            {
                "type": "test_result",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
            }
        )
        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
            return None
        self._print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            self._print(f"  • {failure}")
        self._restate_goal(test_result)
        return None

    async def _test(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Ask the tester for a verdict; a tester blow-up is a failed test, not a crash."""
        try:
            with Status("[bold yellow]Testing...", console=self.console):
                tester = TesterAgent(client=self.client, settings=self.settings)
                return await tester.run(goal, tasks)
        except Exception as exc:
            message = f"Tester failed: {exc}"
            self._error(message)
            return TestResult(passed=False, summary=message, failures=[message])

    # ----------------------------------------------------------------------- state

    def _merge_tasks(self, planned: list[TaskModel]) -> None:
        """Fold a fresh plan into :attr:`tasks`, keeping the work already finished.

        Blindly replacing the list threw away every result of the previous cycle and
        re-ran work that had already succeeded.  Only genuinely new tasks and tasks
        that previously *failed* are (re)scheduled here; a COMPLETED task keeps the
        exact record its worker's summary describes.
        """
        previous = {task.id: task for task in self.tasks}
        merged: list[TaskModel] = []
        seen: set[str] = set()
        for fresh in planned:
            if fresh.id in seen:
                self._print(
                    f"[yellow]Plan repeats task id {fresh.id}; keeping the first copy.[/yellow]"
                )
                continue
            seen.add(fresh.id)
            prior = previous.get(fresh.id)
            if prior is None:
                # The planner never reports a status; assume work still to do.
                fresh.status = TaskStatus.PENDING
                fresh.result = None
                merged.append(fresh)
                continue
            if prior.status == TaskStatus.COMPLETED:
                merged.append(prior)
                continue
            # Pending or failed: retry it with the planner's freshest wording.
            prior.description = fresh.description
            prior.files = list(fresh.files)
            prior.dependencies = list(fresh.dependencies)
            prior.status = TaskStatus.PENDING
            prior.result = None
            merged.append(prior)
        self.tasks = merged

    def _get_tasks(self) -> list[TaskModel]:
        """The authoritative task list (a method for backwards compatibility)."""
        return self.tasks

    def _is_done(self) -> bool:
        """True only when the goal is *proven*: work finished and the tests say so.

        The previous version compared ``completed >= len(self._get_tasks())`` against
        an always-empty list, so ``0 >= 0`` was true on the very first call and
        ``run()`` printed "Goal complete. Halting." after exactly one cycle.  It also
        ignored ``test_result``, which is computed every cycle and then thrown away -
        so it would have declared success over a red test suite.  Halting while tests
        fail is precisely that bug: the entire point of the next cycle is to fix them.

        Returns:
            True when there is at least one task, every task is COMPLETED, none is
            FAILED, and the most recent test result exists and passed.
        """
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(task.status == TaskStatus.FAILED for task in tasks):
            return False
        if not all(task.status == TaskStatus.COMPLETED for task in tasks):
            return False
        return self.test_result is not None and self.test_result.passed

    def _state_token(self) -> StateToken:
        """Fingerprint of everything a cycle can change, for stall detection."""
        completed = sum(1 for task in self.tasks if task.status == TaskStatus.COMPLETED)
        failed = sum(1 for task in self.tasks if task.status == TaskStatus.FAILED)
        tests = (
            None
            if self.test_result is None
            else (self.test_result.passed, tuple(self.test_result.failures))
        )
        return completed, failed, tests

    def _register_cycle_state(self) -> bool:
        """Update the stall counter; return True once the loop is declared stuck."""
        token = self._state_token()
        self._stalled_cycles = 0 if token != self._last_state else self._stalled_cycles + 1
        self._last_state = token
        return self._stalled_cycles >= STALL_LIMIT

    # ------------------------------------------------------------------ execution

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Run *tasks* level by level, at most ``max_parallel_tasks`` at a time.

        Parallelism is throttled with a semaphore rather than enforced by truncating
        the task list: dropping the tail of a plan would silently lose planned work,
        which is exactly the quiet data loss the persistent task list exists to stop.
        """
        limit = max(1, self.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(limit)
        self._print(
            f"[dim]Executing {len(tasks)} task(s), at most {limit} in parallel.[/dim]"
        )
        levels, unordered = self._dependency_levels(tasks)
        known = {task.id for task in tasks}
        completed_ids = {task.id for task in tasks if task.status == TaskStatus.COMPLETED}
        for level in levels:
            runnable: list[TaskModel] = []
            for task in level:
                if task.status != TaskStatus.PENDING:
                    continue
                unmet = [
                    dep
                    for dep in task.dependencies
                    if dep in known and dep not in completed_ids
                ]
                if unmet and task.id not in unordered:
                    # A dependency really did fail.  Leave the task PENDING rather
                    # than building on a broken foundation; the next cycle retries it.
                    self._print(
                        f"[yellow]Skipping task {task.id}: dependency "
                        f"{', '.join(unmet)} did not complete. Retrying next cycle.[/yellow]"
                    )
                    continue
                runnable.append(task)
            await self._run_level(runnable, semaphore)
            completed_ids.update(
                task.id for task in runnable if task.status == TaskStatus.COMPLETED
            )

    async def _run_level(
        self, level: list[TaskModel], semaphore: asyncio.Semaphore
    ) -> None:
        """Run one dependency level concurrently and record each task's outcome."""
        if not level:
            return
        coroutines: list[Coroutine[Any, Any, str]] = []
        try:
            for task in level:
                coroutines.append(self._run_task(task, semaphore))
        except BaseException:
            # Never leave a created-but-unawaited coroutine behind: Python warns
            # loudly and the level would be one task short for the rest of the run.
            for coroutine in coroutines:
                coroutine.close()
            raise
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for task, result in zip(level, results, strict=True):
            self._record_result(task, result)

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        """Run one worker under the parallelism cap and return its prose summary."""
        async with semaphore:
            self._emit({"type": "task_start", "task_id": task.id})
            self._print(f"[blue]→ {task.id}: {task.description}[/blue]")
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            return await worker.run()

    def _record_result(self, task: TaskModel, result: str | BaseException) -> None:
        """Store a gathered result on its task and publish the matching event."""
        if isinstance(result, BaseException):
            # ``asyncio.gather(..., return_exceptions=True)`` returns a
            # ``CancelledError`` - a BaseException on 3.10+ - for a cancelled task.
            # Testing ``Exception`` instead would file that as a worker's summary.
            task.status = TaskStatus.FAILED
            task.result = f"{type(result).__name__}: {result}"
            self._emit({"type": "task_failed", "task_id": task.id, "error": task.result})
            self._print(f"[red]Task {task.id} failed: {result}[/red]")
            return
        task.status = TaskStatus.COMPLETED
        task.result = result
        self._emit({"type": "task_complete", "task_id": task.id, "summary": result})
        self._print(f"[green]Task {task.id} completed[/green]")

    def _dependency_levels(
        self, tasks: list[TaskModel]
    ) -> tuple[list[list[TaskModel]], set[str]]:
        """Layer *tasks* topologically, and report the ids that cannot be layered.

        A task lands one level below its deepest dependency, so level N is only
        reached once everything it depends on has run - that is what
        ``TaskModel.dependencies`` promises, and nothing read it before.  The
        relaxation is capped at ``len(tasks) + 1`` passes because a dependency cycle
        would otherwise push the levels up forever; the cap degrades a cycle into
        "everything still runs, at the end" instead of dropping work.

        Args:
            tasks: The task list to layer.

        Returns:
            ``(levels, unordered)`` - the levels in execution order, and the ids
            whose dependencies can never all be COMPLETED (a cycle, or a task
            waiting on one).  The caller treats those as runnable so a broken plan
            can never hang or drop the work.
        """
        if not tasks:
            return [], set()
        known = {task.id for task in tasks}
        # An id the plan does not contain can never become COMPLETED, so it is not a
        # blocker at all - the task is treated as runnable.
        deps: dict[str, list[str]] = {}
        for task in tasks:
            unknown = [dep for dep in task.dependencies if dep not in known]
            if unknown:
                self._print(
                    f"[yellow]Task {task.id} depends on unknown task id(s) "
                    f"{', '.join(unknown)}; treating them as satisfied.[/yellow]"
                )
            deps[task.id] = [dep for dep in task.dependencies if dep in known]

        depth: dict[str, int] = {task.id: 0 for task in tasks}
        for _ in range(len(tasks) + 1):
            changed = False
            for task in tasks:
                wanted = 1 + max((depth[dep] for dep in deps[task.id]), default=-1)
                if wanted > depth[task.id]:
                    depth[task.id] = wanted
                    changed = True
            if not changed:
                break

        # Peel from the leaves: anything still left sits in a cycle (or behind one)
        # and so can never have all of its dependencies COMPLETED.
        pending = set(depth)
        while True:
            peeled = {
                task_id
                for task_id in pending
                if not any(dep in pending for dep in deps[task_id])
            }
            if not peeled:
                break
            pending -= peeled
        if pending:
            self._print(
                f"[yellow]Dependency cycle among task(s) {sorted(pending)}; running "
                "them without waiting on every dependency.[/yellow]"
            )

        levels: dict[int, list[TaskModel]] = {}
        for task in tasks:
            levels.setdefault(depth[task.id], []).append(task)
        return [levels[key] for key in sorted(levels)], pending

    # ------------------------------------------------------------------------ goal

    def _restate_goal(self, test_result: TestResult) -> None:
        """Rebuild the working goal from the original objective plus the failures.

        The previous version did ``self.goal = "Fix failing tests:\\n" + "\\n".join(...)``,
        which on the JSON-parse-failure path (``failures`` empty) produced the
        content-free literal ``"Fix failing tests:\\n"`` and lost the objective
        entirely.  The goal is rebuilt from :attr:`original_goal` every time, so it
        can neither lose the objective nor grow without bound across cycles.
        """
        if test_result.failures:
            detail = "\n".join(f"- {failure}" for failure in test_result.failures)
            body = f"The test suite is failing. Fix these failures:\n{detail}"
        else:
            # A failing verdict with no detail must not become a content-free prompt;
            # say so out loud and hand the objective back to the planner instead.
            body = (
                "The test suite is still failing, but the tester reported no "
                "individual failures. Re-read the goal, check the current code "
                "yourself and fix what is missing.\n"
                f"Tester summary: {test_result.summary or '(none)'}"
            )
        self.goal = f"{self.original_goal}\n\n{body}"
        self._emit({"type": "goal_updated", "goal": self.goal})
        self._print("[yellow]Will attempt a fix in the next cycle.[/yellow]")

    # ---------------------------------------------------------------------- events

    def _emit(self, event: dict[str, Any]) -> None:
        """Append to :attr:`events` and push the event to the optional sink.

        The sink is a WebSocket send in production, so it may raise on a closed
        connection.  That is swallowed on purpose: a listener that disappears must
        not take the run with it.  :attr:`events` is filled either way, so a
        reconnecting client can read the history it missed.
        """
        self.events.append(event)
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception as exc:
            self._print(f"[yellow]on_event sink raised {exc!r}; continuing.[/yellow]")

    def _error(self, message: str) -> None:
        """Report a failure on both channels: the event stream and the console."""
        self._emit({"type": "error", "message": message})
        self._print(f"[red]{message}[/red]")

    def _print(self, message: Any) -> None:
        """Every human-facing line goes through :attr:`console`, and nowhere else.

        Rich writes to a stream, and a stream can be closed (a detached CLI, a
        dropped socket), which must not end the run - the event stream stays
        authoritative.  Injecting ``console`` also keeps stdout free for tests and
        for the web tier.
        """
        try:
            self.console.print(message)
        except Exception:
            pass

    def _finish(self, reason: str) -> None:
        """Publish the terminal ``done`` event and say the same thing on the console."""
        self._emit({"type": "done", "reason": reason, "cycles": self.cycles})
        if reason == HALT_GOAL_COMPLETE:
            self._print("[bold green]Goal complete. Halting.[/bold green]")
            return
        self._print(f"[bold yellow]Halting after {self.cycles} cycle(s): {reason}.[/bold yellow]")
