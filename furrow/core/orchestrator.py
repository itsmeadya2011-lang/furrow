from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs plan -> parallel work -> test cycles until the goal is satisfied.

    ``goal`` is immutable for the lifetime of the orchestrator. When a test run
    fails, the failure text is carried into the *next* planner call through
    ``focus`` so that every banner, panel and emitted event keeps reporting the
    goal the user actually asked for.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[..., Any] | None = None,
    ) -> None:
        self.goal = goal
        # focus == goal means "no special attention"; set to the failure text for
        # exactly one cycle after a failing test run.
        self.focus = goal
        # Resolve settings first: the client must be built FROM them, otherwise a
        # per-request ``model``/``max_parallel_tasks`` override would be ignored by
        # every agent, which all read the client's settings.
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._tests_passed = False

    async def run(self) -> None:
        banner = f"[bold green]Furrow[/bold green]\nGoal: {self.goal}"
        console.print(Panel.fit(banner, title="Furrow"))
        reason = "max_cycles"
        try:
            while self.settings.max_cycles <= 0 or self.cycles < self.settings.max_cycles:
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                done = self._is_done()
                await self._emit("cycle_end", cycle=self.cycles, done=done)
                if not self.tasks:
                    reason = "no_tasks"
                    console.print("[yellow]No tasks planned. Stopping.[/yellow]")
                    break
                if done:
                    reason = "complete"
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
            else:
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]"
                )
        except (asyncio.CancelledError, KeyboardInterrupt):
            await self._emit("done", reason="cancelled")
            console.print("[yellow]Run cancelled.[/yellow]")
            raise
        await self._emit("done", reason=reason)

    async def _cycle(self) -> None:
        await self._emit("cycle_start", cycle=self.cycles, goal=self.goal)
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.focus)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit("plan", cycle=self.cycles, plan=plan.model_dump())

        # The plan is the source of truth for the current cycle; focus only leaks
        # one cycle of failure context.
        self.tasks = list(plan.tasks)
        self.focus = self.goal
        self._tests_passed = False

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(self._guarded(task, sem) for task in self.tasks),
                return_exceptions=True,
            )

        for task, result in zip(self.tasks, results, strict=False):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            elif result is None:
                # Worker could not finish yet; keep the task open for a later cycle.
                task.status = "pending"
                console.print(f"[yellow]Task {task.id} deferred[/yellow]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            await self._emit("task", cycle=self.cycles, task=task.model_dump())

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self.tasks)
        self._tests_passed = test_result.passed
        await self._emit("test", cycle=self.cycles, result=test_result.model_dump())

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.focus = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _guarded(self, task: TaskModel, sem: asyncio.Semaphore) -> Any:
        async with sem:
            return await self._run_worker(task)

    async def _run_worker(self, task: TaskModel) -> str | None:
        return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if any(task.status != "completed" for task in self.tasks):
            return False
        return self._tests_passed

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    async def _emit(self, event_type: str, **payload: Any) -> None:
        """Best-effort, JSON-serialisable event emission for the web UI."""
        if self.on_event is None:
            return
        event = {"type": event_type, **payload}
        try:
            result = self.on_event(event)
            if inspect.isawaitable(result):
                await result
        except Exception:  # a broken UI must never kill the run
            pass
