from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else None)
        if self.settings is None:
            from furrow.config import settings
            self.settings = settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.settings.max_cycles
        safety_cap = max_cycles if (max_cycles and max_cycles > 0) else 50
        while self.cycles < safety_cap:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            continue_running = await self._cycle()
            if not continue_running:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
        else:
            console.print("[yellow]Reached maximum cycles without completion.[/yellow]")

    async def _cycle(self) -> bool:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return False

        self._merge_tasks(plan.tasks)

        max_parallel = self.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_limit(task: TaskModel) -> str:
            async with semaphore:
                agent = WorkerAgent(task=task, client=self.client)
                return await agent.run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [run_with_limit(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            existing = self._find_task(task.id)
            if isinstance(result, Exception):
                existing.status = "failed"
                existing.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                existing.status = "completed"
                existing.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return False

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")
        self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
        return True

    def _find_task(self, task_id: str) -> TaskModel:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task {task_id} not found")

    def _merge_tasks(self, plan_tasks: list[TaskModel]) -> None:
        for plan_task in plan_tasks:
            existing = next((t for t in self.tasks if t.id == plan_task.id), None)
            if existing:
                existing.description = plan_task.description
                existing.files = plan_task.files
                existing.dependencies = plan_task.dependencies
            else:
                self.tasks.append(plan_task)
