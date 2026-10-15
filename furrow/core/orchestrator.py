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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()
_DEFAULT_CONSOLE: Console = console


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.console = console if console is not None else _DEFAULT_CONSOLE
        self.tasks: list[Any] = []
        self._last_test_passed: bool = True

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self.console.print(f"[yellow]Max cycles ({self.client.settings.max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.tasks = plan.tasks   # store tasks from this cycle
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not self.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._last_test_passed = True
            return

        # Determine parallelism limit
        max_parallel = self.client.settings.max_parallel_tasks
        if max_parallel > 0:
            semaphore = asyncio.Semaphore(max_parallel)

            async def run_task_with_semaphore(task):
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()
            tasks_to_run = [run_task_with_semaphore(task) for task in self.tasks]
        else:
            tasks_to_run = [WorkerAgent(task=task, client=self.client).run() for task in self.tasks]

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self._last_test_passed = test_result.passed

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # If tests failed, keep looping (goal gets rewritten for next cycle)
        if not getattr(self, "_last_test_passed", True):
            return False
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if completed + failed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.tasks
