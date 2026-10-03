from __future__ import annotations

import asyncio

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

MAX_CONSECUTIVE_FAILURES = 3
MAX_NO_PROGRESS_CYCLES = 3


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._feedback: str = ""
        self._tests_failed = False
        self._semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        errors = 0
        stalled = 0
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                progressed = await self._cycle()
            except Exception as exc:
                errors += 1
                console.print(f"[red]Cycle failed: {exc}[/red]")
                if errors >= MAX_CONSECUTIVE_FAILURES:
                    console.print(
                        f"[bold red]{errors} consecutive cycle failures. Halting.[/bold red]"
                    )
                    break
                continue
            errors = 0
            stalled = 0 if progressed else stalled + 1

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if stalled >= MAX_NO_PROGRESS_CYCLES:
                console.print(
                    f"[bold red]No tasks completed in {stalled} cycles. Halting.[/bold red]"
                )
                break
            budget = self.settings.max_cycles
            if budget > 0 and self.cycles >= budget:
                console.print(
                    f"[yellow]Cycle budget exhausted ({self.cycles}/{budget}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> bool:
        """Run one plan/execute/test cycle. Returns True if any task completed."""
        planning_goal = f"{self._feedback}\n\n{self.goal}" if self._feedback else self.goal
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(planning_goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = plan.tasks
        if not self._tasks:
            console.print("[yellow]No tasks planned. Nothing to do this cycle.[/yellow]")
            return False

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            await self._execute(self._tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)

        self._tests_failed = not test_result.passed
        if test_result.passed:
            self._feedback = ""
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._feedback = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return any(t.status == "completed" for t in self._tasks)

    async def _execute(self, tasks: list[TaskModel]) -> None:
        """Run tasks in dependency waves, never exceeding settings.max_parallel_tasks."""
        by_id = {t.id: t for t in tasks}
        pending = list(tasks)
        while pending:
            ready: list[TaskModel] = []
            skipped: list[TaskModel] = []
            for task in pending:
                deps = [by_id[d] for d in task.dependencies if d in by_id]
                if any(d.status in ("failed", "skipped") for d in deps):
                    skipped.append(task)
                elif all(d.status == "completed" for d in deps):
                    ready.append(task)
            for task in skipped:
                task.status = "skipped"
                task.result = "skipped: dependency did not complete"
                console.print(f"[yellow]Task {task.id} skipped (dependency failed)[/yellow]")
            if not ready:
                ready = [t for t in pending if t.status != "skipped"]
                console.print("[yellow]Unresolvable dependencies; running rest unordered.[/yellow]")
            await self._run_wave(ready)
            handled = {t.id for t in ready} | {t.id for t in skipped}
            pending = [t for t in pending if t.id not in handled]

    async def _run_wave(self, tasks: list[TaskModel]) -> None:
        async def guarded(task: TaskModel) -> str:
            async with self._semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        results = await asyncio.gather(*(guarded(t) for t in tasks), return_exceptions=True)
        for task, result in zip(tasks, results, strict=True):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(t.status == "failed" for t in tasks):
            return True
        # Every task landed, but the goal only holds once the suite is green.
        return all(t.status == "completed" for t in tasks) and not self._tests_failed

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
