from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None
        self._last_test_passed: bool | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        console.print(f"[dim]Max cycles: {'∞' if not self.settings or self.settings.max_cycles == 0 else self.settings.max_cycles} | Max parallel tasks: {self.settings.max_parallel_tasks if self.settings else 5}[/dim]")
        while True:
            if self.settings and self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles={self.settings.max_cycles}. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            try:
                plan = await self.planner.plan(self.goal)
            except Exception as e:
                console.print(f"[red]Planning failed: {e}. Continuing with empty plan.[/red]")
                from furrow.config import Plan
                plan = Plan(tasks=[], rationale=f"Planning error: {e}")
        self._plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks if self.settings else 5)

        async def run_with_semaphore(task):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._last_test_passed = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            self._last_test_passed = False

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if self._last_test_passed is False:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self._plan is None:
            return []
        return self._plan.tasks
