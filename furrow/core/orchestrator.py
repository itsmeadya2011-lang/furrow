from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, WorkerResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.history: list[Plan] = []
        self.test_result: TestResult | None = None
        self._console = console or globals()["console"]

    async def run(self) -> None:
        c = self._console
        c.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        max_cycles = self.settings.max_cycles if self.settings else 0
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                c.print(
                    f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]"
                )
                break
            self.cycles += 1
            c.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                c.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        c = self._console
        with Status("[bold yellow]Planning...", console=c):
            plan = await self.planner.plan(self.goal)
        c.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            c.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store the current plan's tasks so _is_done can evaluate them.
        self.tasks = plan.tasks
        self.history.append(plan)

        max_parallel = self._max_parallel()
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_worker(task: TaskModel) -> WorkerResult:
            async with semaphore:
                return await WorkerAgent(
                    task=task, client=self.client, settings=self.settings
                ).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=c):
            results = await asyncio.gather(
                *(_run_worker(t) for t in plan.tasks), return_exceptions=True
            )

        for idx, result in enumerate(results):
            task = plan.tasks[idx]
            if isinstance(result, WorkerResult):
                task.status = "completed"
                task.result = result.summary
                if result.files:
                    files_written = ", ".join(f.path for f in result.files)
                    c.print(
                        f"[green]Task {task.id} completed[/green] — wrote: {files_written}"
                    )
                else:
                    c.print(f"[green]Task {task.id} completed[/green]")
            else:
                task.status = "failed"
                task.result = str(result)
                c.print(f"[red]Task {task.id} failed: {result}[/red]")

        with Status("[bold yellow]Testing...", console=c):
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)
        self.test_result = test_result

        if test_result.passed:
            c.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            c.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                c.print(f"  • {failure}")
            c.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _max_parallel(self) -> int:
        """Return the maximum number of parallel worker tasks."""
        if self.settings and self.settings.max_parallel_tasks > 0:
            return self.settings.max_parallel_tasks
        return 5

    def _is_done(self) -> bool:
        """Check if all tasks in the latest plan are completed."""
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed < len(self.tasks):
            return False
        return not (self.test_result is not None and not self.test_result.passed)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
