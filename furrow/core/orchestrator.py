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
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        planner: PlannerAgent | None = None,
        tester: TesterAgent | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = planner or PlannerAgent(client=self.client, settings=self.settings)
        self.tester = tester or TesterAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None
        self.cycles = 0

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while self.settings.max_cycles == 0 or self.cycles < self.settings.max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.tasks = plan.tasks

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*(run_task(task) for task in self.tasks), return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            self.test_result = await self.tester.run(self.goal, self.tasks)

        if self.test_result.passed:
            console.print(f"[green]Tests passed: {self.test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {self.test_result.summary}[/red]")
            for failure in self.test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

            failed_task_details = "\n".join(
                f"- {task.id}: {task.description} (result: {task.result or 'none'})"
                for task in self.tasks
                if task.status == "failed"
            )
            self.goal = (
                "Fix failing tests for the original goal:\n"
                f"Original goal: {self.original_goal}\n"
                "Failed task details:\n"
                f"{failed_task_details or '- None'}\n"
                "Test failures:\n"
                + "\n".join(self.test_result.failures)
            )

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        return (
            bool(tasks)
            and self.test_result is not None
            and self.test_result.passed
            and not any(task.status == "failed" for task in tasks)
            and all(task.status == "completed" for task in tasks)
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
