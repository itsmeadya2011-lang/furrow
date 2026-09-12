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
from furrow.config import Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        planner: PlannerAgent | None = None,
        tester: TesterAgent | None = None,
        max_cycles: int | None = None,
        max_parallel_tasks: int | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        client_settings = (
            settings
            if settings is not None
            else getattr(self.client, "settings", None)
        )
        self.planner = planner or PlannerAgent(client=self.client, settings=settings)
        self.tester = tester or TesterAgent(client=self.client, settings=settings)
        self.cycles = 0
        self.current_tasks: list[Any] = []
        self.last_test_result: TestResult | None = None
        self.failures: list[str] = []
        self.max_cycles = (
            max_cycles
            if max_cycles is not None
            else (client_settings.max_cycles if client_settings is not None else 0)
        )
        self.max_parallel_tasks = (
            max_parallel_tasks
            if max_parallel_tasks is not None
            else (client_settings.max_parallel_tasks if client_settings is not None else 5)
        )
        if self.max_cycles < 0:
            raise ValueError("max_cycles must be >= 0")
        if self.max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be >= 1")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while self.max_cycles <= 0 or self.cycles < self.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if not self.current_tasks:
                console.print("[bold green]No tasks planned. Halting.[/bold green]")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, failures=self.failures)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.current_tasks = plan.tasks
        if not self.current_tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(max(1, self.max_parallel_tasks))

        async def run_task(task: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [asyncio.create_task(run_task(task)) for task in self.current_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.current_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self.current_tasks)

        self.last_test_result = test_result
        if test_result.passed:
            self.failures = []
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.failures = list(test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in self.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    def _is_done(self) -> bool:
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False

        tasks = self._get_tasks()
        completed = sum(1 for task in tasks if task.status == "completed")
        failed = sum(1 for task in tasks if task.status == "failed")
        return failed == 0 and completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self.current_tasks

