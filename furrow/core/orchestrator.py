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
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._all_tasks: list[TaskModel] = []
        self.settings = Settings()

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._merge_tasks(plan.tasks)

        pending_tasks = [t for t in self._all_tasks if t.status == "pending"]
        if not pending_tasks:
            console.print("[yellow]No pending tasks to execute.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
            async with semaphore:
                return task, await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_futures = [run_with_semaphore(task) for task in pending_tasks]
            results = await asyncio.gather(*worker_futures, return_exceptions=True)

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self._all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        existing_ids = {t.id for t in self._all_tasks}
        for task in new_tasks:
            if task.id not in existing_ids:
                self._all_tasks.append(task)
                existing_ids.add(task.id)

    def _is_done(self) -> bool:
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
            return True

        tasks = self._get_tasks()
        if not tasks:
            return True

        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False

        completed = sum(1 for t in tasks if t.status == "completed")
        pending = sum(1 for t in tasks if t.status == "pending")

        if pending > 0:
            return False

        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self._all_tasks
