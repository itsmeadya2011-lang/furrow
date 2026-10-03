from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

DEFAULT_MAX_STALLED_CYCLES = 3


class Orchestrator:
    """Drive the plan -> execute -> test loop toward a goal.

    Every cycle asks :class:`PlannerAgent` for a plan, executes the planned
    :class:`~furrow.config.TaskModel` objects with :class:`WorkerAgent`, then asks
    :class:`TesterAgent` whether the goal holds. Tasks run as soon as their declared
    ``dependencies`` finish, independent tasks run concurrently, and concurrency is
    bounded by ``settings.max_parallel_tasks``.

    Termination policy - the loop halts when any of these is true:

    * the last cycle finished every task with no failures **and** the tests passed,
      which is the only proof of completion (an empty plan is not proof on its own);
    * ``settings.max_cycles`` (> 0; ``<= 0`` means unlimited) cycles have run;
    * the no-progress circuit breaker sees an unchanged task-set signature
      ``max_stalled_cycles`` (3 by default) cycles in a row;
    * a cycle failed so badly that no task state changed - e.g. a planner or tester
      crash - which also shows up as an unchanged signature.

    ``on_event`` receives a plain-text line for every notable state change (cycle
    start, plan, task start/completion/failure, test result, stop reason) so a UI can
    stream progress. Rich markup is used for the console only.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], None] | None = None,
        *,
        max_stalled_cycles: int = DEFAULT_MAX_STALLED_CYCLES,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.last_test_result: TestResult | None = None
        self.on_event = on_event
        self.max_stalled_cycles = max(1, max_stalled_cycles)
        self._last_signature: tuple[int, tuple[str, ...], bool] | None = None
        self._stalled_cycles = 0

    # ------------------------------------------------------------------ loop

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        self._emit(f"Starting Furrow: {self.original_goal}")
        max_cycles = self._max_cycles()
        if max_cycles > 0:
            self._emit(f"Cycle limit is {max_cycles}")

        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._halt(f"Reached the configured cycle limit of {max_cycles} cycle(s).")
                return

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(f"Cycle {self.cycles} started")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("Loop stopping: goal complete")
                return
            if not self._record_progress():
                return

    async def _cycle(self) -> None:
        """Run one plan/execute/test cycle; never raises into :meth:`run`."""
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        except Exception as exc:  # noqa: BLE001 - bad LLM JSON must not kill the run
            self._cycle_error("planning", exc)
            return

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        tasks = self._clamp(plan.tasks)
        self.tasks = tasks
        self._emit(f"Plan ready with {len(tasks)} task(s)")
        if not tasks:
            console.print(
                "[yellow]No tasks planned; asking the tester whether the goal already holds.[/yellow]"
            )

        try:
            await self._execute(tasks)
        except Exception as exc:  # noqa: BLE001 - a scheduler blow-up is a failed cycle
            self._cycle_error("execution", exc)
            return

        await self._test(tasks)

    # -------------------------------------------------------------- planning

    def _clamp(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Bound fan-out to ``settings.max_parallel_tasks``."""
        limit = self._parallel_limit()
        if len(tasks) <= limit:
            return list(tasks)
        dropped = len(tasks) - limit
        message = (
            f"Plan has {len(tasks)} tasks; running the first {limit} "
            f"and deferring {dropped} to a later cycle."
        )
        console.print(f"[yellow]{message}[/yellow]")
        self._emit(message)
        return list(tasks[:limit])

    def _compose_goal(self, test_result: TestResult) -> str:
        """Next cycle's goal: the original objective plus failure detail (never replaces it)."""
        detail = "\n".join(f"- {failure}" for failure in test_result.failures).strip()
        header = f"Original goal: {self.original_goal}"
        if not detail:
            return (
                f"{header}\n\nTests did not pass, but no specific failures were reported. "
                "Re-attempt the original goal and verify the result yourself."
            )
        return (
            f"{header}\n\nThe previous attempt left these failures; fix them while still "
            f"achieving the original goal:\n{detail}"
        )

    # ------------------------------------------------------------- execution

    async def _execute(self, tasks: list[TaskModel]) -> None:
        """Run tasks respecting ``dependencies``, capped by a semaphore."""
        semaphore = asyncio.Semaphore(self._parallel_limit())
        deps = self._resolve_dependencies(tasks)
        queue = self._dedupe(tasks)
        done: set[str] = set()
        failed: set[str] = set()
        running: dict[asyncio.Task[bool], TaskModel] = {}

        while True:
            self._fail_blocked(queue, deps, done, failed)
            if not running:
                ready = [task for task in queue.values() if all(d in done for d in deps[task.id])]
                if not ready and queue:
                    ready = [self._break_cycle(queue, deps)]
                for task in ready:
                    queue.pop(task.id, None)
                    running[asyncio.ensure_future(self._run_task(task, semaphore))] = task
            if not running:
                return

            finished, _ = await asyncio.wait(list(running), return_when=asyncio.FIRST_COMPLETED)
            for future in finished:
                task = running.pop(future)
                done.add(task.id)
                if not future.result():
                    failed.add(task.id)

    def _resolve_dependencies(self, tasks: list[TaskModel]) -> dict[str, list[str]]:
        """Map task id -> runnable dependency ids, warning about dangling ones."""
        known = {task.id for task in tasks}
        deps: dict[str, list[str]] = {}
        for task in tasks:
            resolved = []
            for dep in task.dependencies:
                if dep in known and dep != task.id:
                    resolved.append(dep)
                    continue
                warning = (
                    f"Task {task.id} declares dependency {dep!r}, which is not a distinct "
                    "task in this plan; treating it as independent."
                )
                console.print(f"[yellow]{warning}[/yellow]")
                self._emit(warning)
            deps[task.id] = resolved
        return deps

    def _dedupe(self, tasks: list[TaskModel]) -> dict[str, TaskModel]:
        """Build the runnable queue, rejecting duplicate ids so the graph stays sane."""
        queue: dict[str, TaskModel] = {}
        for task in tasks:
            if task.id in queue:
                task.status = "failed"
                task.result = "duplicate task id in plan"
                message = f"Task {task.id} has a duplicate id in the plan; dropping the repeat."
                console.print(f"[red]{message}[/red]")
                self._emit(message)
                continue
            queue[task.id] = task
        return queue

    def _fail_blocked(
        self,
        queue: dict[str, TaskModel],
        deps: dict[str, list[str]],
        done: set[str],
        failed: set[str],
    ) -> None:
        """Mark tasks whose dependencies all resolved but at least one failed."""
        changed = True
        while changed:
            changed = False
            for task in list(queue.values()):
                task_deps = deps[task.id]
                if not all(dep in done for dep in task_deps):
                    continue
                broken = [dep for dep in task_deps if dep in failed]
                if not broken:
                    continue
                queue.pop(task.id, None)
                done.add(task.id)
                failed.add(task.id)
                task.status = "failed"
                task.result = f"blocked by failed dependency: {', '.join(broken)}"
                message = f"Task {task.id} skipped; dependency {', '.join(broken)} failed."
                console.print(f"[red]{message}[/red]")
                self._emit(message)
                changed = True

    def _break_cycle(self, queue: dict[str, TaskModel], deps: dict[str, list[str]]) -> list[TaskModel]:
        """Nothing is ready, so the remaining tasks form a dependency cycle: cut it."""
        stuck = sorted(task.id for task in queue.values() if deps[task.id])
        chosen = min(queue.values(), key=lambda task: len(deps[task.id]))
        warning = (
            f"Dependency cycle detected among tasks {', '.join(stuck)}; "
            f"breaking it by running {chosen.id} first."
        )
        console.print(f"[yellow]{warning}[/yellow]")
        self._emit(warning)
        return [chosen]

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> bool:
        async with semaphore:
            task.status = "running"
            self._emit(f"Task {task.id} started: {task.description}")
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
            except Exception as exc:  # noqa: BLE001 - one bad task must not abort the cycle
                task.status = "failed"
                task.result = str(exc)
                console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                self._emit(f"Task {task.id} failed: {exc}")
                return False
            task.status = "completed"
            task.result = result
            console.print(f"[green]Task {task.id} completed[/green]")
            self._emit(f"Task {task.id} completed")
            return True

    # --------------------------------------------------------------- testing

    async def _test(self, tasks: list[TaskModel]) -> None:
        try:
            with Status("[bold yellow]Testing...", console=console):
                test_result = await TesterAgent(client=self.client).run(self.goal, tasks)
        except Exception as exc:  # noqa: BLE001 - a tester crash is a failed cycle
            self.last_test_result = TestResult(
                passed=False, summary=f"Tester crashed: {exc}", failures=[str(exc)]
            )
            self._cycle_error("testing", exc)
            self.goal = self._compose_goal(self.last_test_result)
            return

        self.last_test_result = test_result
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"Tests passed: {test_result.summary}")
            return

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")
        console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        self._emit(f"Tests failed: {test_result.summary}")
        self.goal = self._compose_goal(test_result)

    # ----------------------------------------------------------- termination

    def _is_done(self) -> bool:
        """True only when the last cycle finished cleanly and the tests passed."""
        tasks = self._get_tasks()
        if any(task.status == "failed" for task in tasks):
            return False
        if any(task.status != "completed" for task in tasks):
            return False
        if self.last_test_result is None:
            return False
        return self.last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _record_progress(self) -> bool:
        """Track the task-set signature; return False once the loop has stalled."""
        signature = self._signature()
        if signature == self._last_signature:
            self._stalled_cycles += 1
        else:
            self._last_signature = signature
            self._stalled_cycles = 1
        if self._stalled_cycles < self.max_stalled_cycles:
            return True
        self._halt(
            f"No progress after {self._stalled_cycles} cycles "
            f"(same plan and task state, tests still failing); stopping."
        )
        return False

    def _signature(self) -> tuple[int, tuple[str, ...], bool]:
        return (
            len(self.tasks),
            tuple(task.status for task in self.tasks),
            bool(self.last_test_result and self.last_test_result.passed),
        )

    def _max_cycles(self) -> int:
        return int(self.client.settings.max_cycles)

    def _parallel_limit(self) -> int:
        """Concurrency budget, never below 1 so the semaphore cannot deadlock."""
        return max(1, int(self.client.settings.max_parallel_tasks))

    def _halt(self, reason: str) -> None:
        console.print(f"[bold yellow]{reason}[/bold yellow]")
        self._emit(f"Loop stopping: {reason}")

    def _cycle_error(self, stage: str, error: Exception) -> None:
        message = f"Cycle {self.cycles} {stage} failed: {error}"
        console.print(f"[red]{message}[/red]")
        self._emit(message)

    def _emit(self, message: str) -> None:
        """Send a single-line plain-text event to ``on_event``; a bad sink is dropped."""
        if self.on_event is None:
            return
        flat = " ".join(str(message).split())
        try:
            self.on_event(flat)
        except Exception as exc:  # noqa: BLE001 - a broken UI sink must not kill the loop
            self.on_event = None
            console.print(f"[yellow]Event callback disabled after error: {exc}[/yellow]")
