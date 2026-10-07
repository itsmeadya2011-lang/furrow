from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal: str = goal
        self.client: LLMClient = client or LLMClient()
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles: int = 0
        self.tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None
        self.last_test_passed: bool = False

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
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._merge_tasks(plan.tasks)

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            id_to_task: dict[str, TaskModel] = {t.id: t for t in self.tasks}
            tasks = [
                WorkerAgent(task=id_to_task[t.id], client=self.client).run()
                for t in plan.tasks
                if t.id in id_to_task
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        id_to_task = {t.id: t for t in self.tasks}
        for plan_task, result in zip(plan.tasks, results):
            task = id_to_task[plan_task.id]
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            self.test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if self.test_result.passed:
            self.last_test_passed = True
            console.print(f"[green]Tests passed: {self.test_result.summary}[/green]")
        else:
            self.last_test_passed = False
            console.print(f"[red]Tests failed: {self.test_result.summary}[/red]")
            for failure in self.test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self.test_result.failures)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        existing_by_id: dict[str, TaskModel] = {t.id: t for t in self.tasks}
        for new_task in new_tasks:
            existing = existing_by_id.get(new_task.id)
            if existing is None:
                self.tasks.append(new_task)
            else:
                existing.description = new_task.description
                existing.files = new_task.files
                existing.dependencies = new_task.dependencies

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        max_cycles = self.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        if all(t.status == "completed" for t in tasks):
            return self.last_test_passed
        return False
