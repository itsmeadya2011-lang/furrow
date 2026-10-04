from __future__ import annotations

import asyncio
import inspect
import logging

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent, WorkerResult
from furrow.config import Plan, Settings, TaskModel, TaskStatus
from furrow.config import settings as default_settings  # noqa: I001 - module default instance
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)
console = Console()

# A cycle that raises (planner ValueError, LLM/auth error, tester crash, ...) is a "hard
# failure"; a cycle whose test run is red makes no progress either. Consecutive cycles
# that do not progress almost always share one root cause, so the loop gives up after
# this many of them instead of burning credits in a crash/failure loop.
MAX_CONSECUTIVE_CYCLE_FAILURES = 3


class Orchestrator:
    """Runs the plan -> execute -> test loop until the goal is met or a limit is hit."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.original_goal = goal
        # `goal` is never reassigned after __init__: failures from the previous test run are
        # fed forward through `plan_context` so the objective survives every cycle.
        self.goal = goal
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.current_tasks: list[TaskModel] = []
        self.plan_context: str | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._planner_takes_context: bool | None = None
        self._tests_passed: bool | None = None
        self._hard_failure = False
        # Consecutive cycles that made no progress: a hard failure or a red test run.
        self._consecutive_non_progress = 0

    async def run(self) -> None:
        """Run cycles until the goal is complete, a limit is hit, or the loop is stopped.

        Terminates when every planned task completed and the test run passed, when
        `settings.max_cycles` (0 = unlimited) is reached, or after
        `settings.fail_fast_cycles` cycles in a row made no progress -- i.e. raised an
        unexpected exception or finished with a failing test run. A cycle that planned
        no tasks is complete, so it neither counts as non-progressing nor blocks the
        halt. `KeyboardInterrupt`/`CancelledError` are *not* swallowed.
        """
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\n"
                f"Goal: {self.goal}\n"
                f"Workspace: {self.settings.workspace}",
                title="Furrow",
            )
        )
        logger.info(
            "Starting loop (workspace=%s, max_cycles=%s, max_parallel_tasks=%s)",
            self.settings.workspace,
            self.settings.max_cycles,
            self.settings.max_parallel_tasks,
        )
        # Bind the semaphore to the loop that actually runs the workers.
        self._semaphore = None
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as exc:  # noqa: BLE001 - error boundary, see module docstring
                self._hard_failure = True
                self._consecutive_non_progress += 1
                logger.exception("Cycle %s aborted: %s", self.cycles, exc)
                console.print(f"[red]Cycle {self.cycles} aborted: {exc}[/red]")
            else:
                if self._tests_passed is False:
                    # A red test run is no progress either: charge it to the same budget as
                    # a hard failure, otherwise a persistently failing suite loops forever
                    # under the default `max_cycles=0` (unlimited).
                    self._consecutive_non_progress += 1
                else:
                    # Passing tests -- or no tests at all, since a cycle that planned
                    # nothing is complete (see `_is_done`) -- clear the streak.
                    self._consecutive_non_progress = 0

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            reason = self._stop_reason()
            if reason:
                console.print(f"[bold yellow]{reason} Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        """Plan, execute (in dependency waves) and test once."""
        self._hard_failure = False
        self._tests_passed = None
        self.current_tasks = []

        with Status("[bold yellow]Planning...", console=console) as status:
            status.update(f"[bold yellow]Planning cycle {self.cycles}...")
            plan = await self._plan()
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.current_tasks = plan.tasks
        with Status("[bold yellow]Executing tasks...", console=console) as status:
            status.update(f"[bold yellow]Executing {len(self.current_tasks)} tasks...")
            await self._execute(self.current_tasks, status=status)

        tester = TesterAgent(client=self.client, settings=self.settings)
        with Status("[bold yellow]Testing...", console=console) as status:
            status.update(f"[bold yellow]Testing cycle {self.cycles}...")
            test_result = await tester.run(self.goal, self.current_tasks)
        self._tests_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.plan_context = None
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}", markup=False)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.plan_context = self._failure_context(test_result.failures)

    async def _plan(self) -> Plan:
        """Ask the planner for this cycle's plan, forwarding the previous failures."""
        if not self.plan_context:
            return await self.planner.plan(self.goal)
        if self._planner_accepts_context():
            return await self.planner.plan(self.goal, self.plan_context)
        logger.debug("Planner exposes no context argument; forwarding failures inline")
        return await self.planner.plan(f"{self.goal}\n\n{self.plan_context}")

    def _failure_context(self, failures: list[str]) -> str:
        """Build the planning context handed to the next cycle (goal itself is untouched)."""
        if not failures:
            return "The previous attempt did not pass its tests. Re-check the goal."
        listed = "\n".join(f"- {failure}" for failure in failures)
        return f"The previous attempt failed its tests. Fix these failures:\n{listed}"

    async def _execute(self, tasks: list[TaskModel], status: Status | None = None) -> None:
        """Run tasks in dependency-aware waves, at most `max_parallel_tasks` at a time."""
        pending = list(tasks)
        known = {task.id for task in tasks}
        completed: set[str] = set()

        while pending:
            wave, pending = self._take_wave(pending, known, completed)
            if not wave:
                if not pending:
                    break
                # Nothing is runnable but work remains: a dependency cycle. Running the
                # remainder together is better than deadlocking (the workers may well
                # resolve the ordering themselves).
                logger.warning("Dependency cycle among %d tasks; running together", len(pending))
                wave, pending = pending, []

            if status is not None:
                status.update(f"[bold yellow]Executing {len(completed)}/{len(tasks)} tasks...")
            await self._run_wave(wave, status=status)
            completed.update(task.id for task in wave if task.status == "completed")

        if status is not None:
            status.update(f"[bold yellow]Executed {len(completed)}/{len(tasks)} tasks.")
        console.print(
            f"[bold cyan]Wave results:[/bold cyan] {len(completed)}/{len(tasks)} task(s) completed"
        )

    def _take_wave(
        self, pending: list[TaskModel], known: set[str], completed: set[str]
    ) -> tuple[list[TaskModel], list[TaskModel]]:
        """Split `pending` into (runnable now, still waiting).

        A task is runnable once every dependency it declares that exists in the plan is
        `completed`. Unknown dependency ids are treated as satisfied (the planner should
        not be able to strand work), and tasks whose dependencies can never complete are
        marked skipped instead of blocking the cycle.
        """
        pending_ids = {task.id for task in pending}
        dead = known - pending_ids - completed
        runnable: list[TaskModel] = []
        waiting: list[TaskModel] = []

        for task in pending:
            deps = {dep for dep in (task.dependencies or []) if dep in known}
            unmet = deps - completed
            if not unmet:
                runnable.append(task)
            elif unmet <= dead:
                logger.info("Skipping task %s: dependency never completed", task.id)
                task.status = TaskStatus.SKIPPED
                task.result = "Skipped: dependencies did not complete."
                console.print(f"[yellow]Task {task.id} skipped (dependency failed)[/yellow]")
            else:
                waiting.append(task)

        return runnable, waiting

    async def _run_wave(self, wave: list[TaskModel], status: Status | None = None) -> None:
        semaphore = self._get_semaphore()
        if status is not None:
            status.update(f"[bold yellow]Running {len(wave)} task(s)...")
        results = await asyncio.gather(
            *(self._run_task(task, semaphore) for task in wave),
            return_exceptions=True,
        )

        for task, result in zip(wave, results, strict=True):
            if isinstance(result, BaseException):
                task.status = TaskStatus.FAILED
                task.result = str(result)
                logger.warning("Worker for task %s raised: %s", task.id, result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            elif result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result.summary
                console.print(f"[green]Task {task.id} completed: {result.summary}[/green]")
            else:
                task.status = TaskStatus.FAILED
                task.result = result.error or result.summary
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> WorkerResult:
        """Run one worker, bounded by the parallelism semaphore."""
        async with semaphore:
            task.status = TaskStatus.RUNNING
            logger.debug("Running task %s: %s", task.id, task.description)
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            result = await worker.run()
        if not isinstance(result, WorkerResult):
            # Defensive: tolerate a worker that still returns a plain string summary.
            return WorkerResult(success=bool(result), summary=str(result))
        return result

    def _is_done(self) -> bool:
        """True when this cycle's tasks all completed and the test run passed.

        A cycle that planned nothing ran no tests, so `_tests_passed` stays None and
        does not block completion; a cycle whose tests *failed* is never complete, no
        matter how cleanly the tasks finished -- otherwise the loop would report
        "Goal complete" over a red test run.
        """
        tasks = self._get_tasks()
        if self._hard_failure:
            return False
        if self._tests_passed is False:
            return False
        if any(task.status == "failed" for task in tasks):
            return False
        completed = sum(1 for task in tasks if task.status == "completed")
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        """Tasks planned by the current cycle."""
        return self.current_tasks

    def _stop_reason(self) -> str | None:
        """Reason to halt the loop, or None to keep going."""
        max_cycles = self.settings.max_cycles
        if max_cycles and self.cycles >= max_cycles:
            return f"Reached max_cycles ({max_cycles})."
        limit = self._fail_fast_cycles()
        if self._consecutive_non_progress >= limit:
            cause = "hard failure" if self._hard_failure else "failing test run"
            return f"{limit} consecutive cycles without progress ({cause}, fail_fast_cycles)."
        return None

    def _fail_fast_cycles(self) -> int:
        """Consecutive non-progressing cycles tolerated before halting.

        `settings.fail_fast_cycles` is authoritative (see the `furrow.config` module
        docstring); the module constant is only the fallback for a settings object that
        predates the field. Non-positive values are clamped to 1 rather than honoured:
        the limit is compared with `>=`, so 0 (or a negative number) would halt the loop
        before the first failure -- even right after a *successful* cycle with work left
        -- and switch the safety net off instead of tuning it.
        """
        limit = getattr(self.settings, "fail_fast_cycles", MAX_CONSECUTIVE_CYCLE_FAILURES)
        return max(1, int(limit))

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily create the worker semaphore so it binds to the running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        return self._semaphore

    def _planner_accepts_context(self) -> bool:
        """Whether `PlannerAgent.plan` takes an optional extra context argument."""
        if self._planner_takes_context is None:
            try:
                params = inspect.signature(self.planner.plan).parameters.values()
            except (TypeError, ValueError):  # pragma: no cover - exotic callables
                self._planner_takes_context = False
            else:
                positional = (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                self._planner_takes_context = (
                    sum(1 for param in params if param.kind in positional) >= 2
                )
        return self._planner_takes_context
