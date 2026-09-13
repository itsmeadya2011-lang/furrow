from __future__ import annotations

import asyncio
from typing import TypeAlias

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, settings, TestResult
from furrow.llm import LLMClient

console = Console()
TaskResult: TypeAlias = str | BaseException


class Orchestrator:
    def __init__(
        self, goal: str, client: LLMClient | None = None, config: Settings | None = None
    ) -> None:
        self.goal = goal
        self.settings = config or settings
        self.max_cycles = self.settings.max_cycles
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._should_stop():
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                else:
                    console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self._plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            if self.last_test_result is not None and not self.last_test_result.passed:
                return
            self.tasks = []
            return

        self.tasks = plan.tasks

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await self._execute_tasks(self.tasks)

        for task, result in zip(self.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self._test(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self.last_test_result = test_result

    async def _plan(self, goal: str) -> Plan:
        return await self.planner.plan(goal)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[TaskResult]:
        worker_tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in tasks
        ]
        return await asyncio.gather(*worker_tasks, return_exceptions=True)

    async def _test(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return await TesterAgent(client=self.client).run(goal, tasks)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            if self.last_test_result is not None and not self.last_test_result.passed:
                return False
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        if completed >= len(tasks):
            if self.last_test_result is not None and not self.last_test_result.passed:
                return False
            return True
        return False

    def _should_stop(self) -> bool:
        if self._is_done():
            return True
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
