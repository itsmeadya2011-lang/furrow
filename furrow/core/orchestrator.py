from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TaskStatus, TestResult, get_settings
from furrow.llm import LLMClient

console = Console()

# How many back-to-back failing cycles are tolerated when ``settings.max_cycles``
# is unlimited (``<= 0``). Without this bound a planner that never converges would
# loop forever.
MAX_CONSECUTIVE_FAILURES = 3


class Orchestrator:
    """Drives the plan -> execute -> test cycle until the goal is met."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0
        self.plan: Plan | None = None
        self.test_result: TestResult | None = None
        self._repair_context: str | None = None
        self._consecutive_failures: int = 0
        self.stopped: bool = False

    def _emit(self, event_type: str, **fields: Any) -> None:
        """Send a progress event to ``on_event``.

        The callback is user supplied (the web UI), so any failure it raises is
        swallowed: a broken callback must never break the development loop.
        """
        if self.on_event is None:
            return
        event: dict[str, Any] = {"type": event_type, "cycle": self.cycles, **fields}
        try:
            self.on_event(event)
        except Exception:
            pass

    async def run(self) -> None:
        """Loop plan/execute/test until a real stop condition is reached."""
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        self._emit("started", goal=self.goal)
        max_cycles = get_settings().max_cycles
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle")
            cycle_ok = False
            try:
                cycle_ok = await self._cycle()
            except Exception as exc:
                # A broken cycle counts as a failed cycle, not a reason to give up.
                console.print(f"[red]Cycle {self.cycles} aborted: {exc!r}[/red]")
                self.plan = None
                self.test_result = None
                self._emit("error", message=str(exc), error=repr(exc))
            if cycle_ok:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
            reason = self._stop_reason(max_cycles)
            if reason is not None:
                self._halt(reason)
                return

    def _stop_reason(self, max_cycles: int) -> str | None:
        """Why the loop must halt, or ``None`` to keep looping."""
        if self.plan is not None and not self.plan.tasks:
            return "Goal complete. Halting. (planner reported no remaining tasks)"
        if self._is_done() and self.test_result is not None and self.test_result.passed:
            return "Goal complete. Halting."
        if self._consecutive_failures > MAX_CONSECUTIVE_FAILURES:
            return (
                f"Stopping: {self._consecutive_failures} consecutive failing cycles "
                f"exceeded the limit of {MAX_CONSECUTIVE_FAILURES}."
            )
        if max_cycles > 0 and self.cycles >= max_cycles:
            return f"Stopping: reached the configured cycle limit ({max_cycles})."
        return None

    def _halt(self, reason: str) -> None:
        """Mark the run as stopped and report why."""
        self.stopped = True
        style = "bold green" if reason.startswith("Goal complete") else "bold yellow"
        console.print(f"[{style}]{reason}[/{style}]")
        self._emit("stopped", reason=reason)

    def _planner_goal(self) -> str:
        """The immutable goal plus the previous cycle's repair context, if any."""
        if self._repair_context:
            return f"{self.goal}\n\n{self._repair_context}"
        return self.goal

    async def _cycle(self) -> bool:
        """Plan, execute and test once. ``True`` means the cycle was fully clean."""
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self._planner_goal())
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit("plan", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return True

        levels, blocked = self._schedule_levels(plan.tasks)
        self._print_schedule(levels, blocked)
        for task, reason in blocked:
            task.status = TaskStatus.FAILED
            task.result = reason
            console.print(f"[red]Task {task.id} skipped: {reason}[/red]")
            self._emit("task", id=task.id, status=task.status.value, result=reason)

        await self._execute(levels)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.test_result = test_result
        self._emit("tests", passed=test_result.passed, summary=test_result.summary)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._repair_context = None
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._repair_context = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return self._is_done() and test_result.passed

    def _schedule_levels(
        self, tasks: list[TaskModel]
    ) -> tuple[list[list[TaskModel]], list[tuple[TaskModel, str]]]:
        """Group tasks into dependency levels.

        A task is only scheduled once every one of its dependencies has completed,
        and all tasks inside a level run concurrently. Returns ``(levels, blocked)``
        where ``blocked`` pairs every task that can never run with the reason why:
        an unknown dependency id, a self dependency, or a dependency cycle.
        """
        known = {task.id for task in tasks}
        blocked: list[tuple[TaskModel, str]] = []
        pending: list[TaskModel] = []
        for task in tasks:
            unknown = sorted({dep for dep in task.dependencies if dep not in known})
            if unknown:
                blocked.append((task, f"unknown dependencies: {', '.join(unknown)}"))
            elif task.id in task.dependencies:
                blocked.append((task, f"depends on itself ({task.id})"))
            else:
                pending.append(task)

        blocked_ids = {task.id for task, _ in blocked}
        pending = [task for task in pending if task.id not in blocked_ids]
        completed: set[str] = set()
        levels: list[list[TaskModel]] = []
        while pending:
            level = [
                task
                for task in pending
                if all(dep in completed for dep in task.dependencies)
            ]
            if not level:
                # Nothing can run any more: whatever is left sits in a cycle or
                # waits on a dependency that was itself blocked.
                for task in pending:
                    unresolved = sorted(
                        dep for dep in task.dependencies if dep not in completed
                    )
                    blocked.append(
                        (task, f"dependency cycle, never runnable: {', '.join(unresolved)}")
                    )
                break
            levels.append(level)
            completed.update(task.id for task in level)
            level_ids = {task.id for task in level}
            pending = [task for task in pending if task.id not in level_ids]
        return levels, blocked

    def _print_schedule(
        self, levels: list[list[TaskModel]], blocked: list[tuple[TaskModel, str]]
    ) -> None:
        """Show how tasks were grouped by dependency."""
        console.print("[bold]Dependency schedule:[/bold]")
        for number, level in enumerate(levels, start=1):
            ids = ", ".join(task.id for task in level)
            console.print(f"  [cyan]level {number}[/cyan]: {ids}")
        for task, reason in blocked:
            console.print(f"  [red]blocked[/red]: {task.id} ({reason})")

    async def _execute(self, levels: list[list[TaskModel]]) -> None:
        """Run each dependency level in order, bounded by ``max_parallel_tasks``."""
        settings = get_settings()
        semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tasks))
        for number, level in enumerate(levels, start=1):
            console.print(
                f"[bold yellow]Executing level {number}/{len(levels)} "
                f"({len(level)} task(s))...[/bold yellow]"
            )
            with Status("[bold yellow]Working...", console=console):
                results = await asyncio.gather(
                    *(self._run_task(task, semaphore) for task in level),
                    return_exceptions=True,
                )
            for task, result in zip(level, results):
                if isinstance(result, BaseException):
                    task.status = TaskStatus.FAILED
                    task.result = f"{type(result).__name__}: {result}"
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                self._emit(
                    "task", id=task.id, status=task.status.value, result=task.result
                )

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        """Run one worker, waiting for a slot in the parallelism budget."""
        async with semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(task.status == TaskStatus.FAILED for task in tasks):
            return False
        return all(task.status == TaskStatus.COMPLETED for task in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.plan.tasks if self.plan is not None else []