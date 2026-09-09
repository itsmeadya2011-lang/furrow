from __future__ import annotations

import asyncio

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
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        if settings is not None:
            self.settings = settings
            if client is not None:
                client.settings = settings
                self.client = client
            else:
                self.client = LLMClient(settings=settings)
        else:
            self.client = client or LLMClient()
            self.settings = self.client.settings
        self.goal = goal
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[bold red]Max cycles ({self.settings.max_cycles}) reached. Goal not complete.[/bold red]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            new_plan = await self.planner.plan(self.goal)

        if self.current_plan is not None:
            task_status_map = {
                task.id: (task.status, task.result) for task in self.current_plan.tasks
            }
            for task in new_plan.tasks:
                if task.id in task_status_map:
                    prev_status, prev_result = task_status_map[task.id]
                    if prev_status == "completed":
                        task.status = "completed"
                        task.result = prev_result
                    else:
                        task.status = "pending"
                        task.result = None

        self.current_plan = new_plan
        console.print(Panel(Pretty(self.current_plan.model_dump()), title="Plan", border_style="blue"))

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks_to_run = [
                WorkerAgent(task=task, client=self.client).run()
                for task in self.current_plan.tasks
                if task.status == "pending"
            ]
            tasks_to_run_ids = [
                task.id for task in self.current_plan.tasks if task.status == "pending"
            ]
            if tasks_to_run:
                results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
                for task_id, result in zip(tasks_to_run_ids, results):
                    task = next(t for t in self.current_plan.tasks if t.id == task_id)
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            self.last_test_result = await TesterAgent(client=self.client).run(self.goal, self.current_plan.tasks)

        if self.last_test_result.passed:
            console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            for task in self.current_plan.tasks:
                task.status = "pending"
                task.result = None
            self.goal = f"Fix failing tests:\n" + "\n".join(self.last_test_result.failures)

    def _is_done(self) -> bool:
        if self.current_plan is None:
            return False

        tasks = self._get_tasks()
        if not tasks:
            if self.last_test_result is not None and self.last_test_result.passed:
                return True
            return False

        if any(t.status != "completed" for t in tasks):
            return False

        if self.last_test_result is None:
            return False

        return self.last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks