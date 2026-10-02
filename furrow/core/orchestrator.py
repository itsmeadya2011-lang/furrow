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
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self.history: list[Plan] = []
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as e:  # noqa: BLE001 - guard against repeated planning failures
                console.print(f"[red]Cycle failed unexpectedly: {e}[/red]")
                break
            tests_done = self.last_test_result is None or self.last_test_result.passed
            if self._is_done() and tests_done:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
        self._print_summary()

    async def _cycle(self) -> TestResult | None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.history.append(plan)
        self._tasks = plan.tasks
        self.last_test_result = None

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return None

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(run_task(t) for t in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        completed = [t for t in self._tasks if t.status == "completed"]
        if not completed:
            console.print("[yellow]No tasks completed successfully. Skipping tests.[/yellow]")
            return None

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self._tasks)
        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        return test_result

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = any(t.status == "failed" for t in tasks)
        completed = all(t.status == "completed" for t in tasks)
        return completed and not failed

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks

    def _print_summary(self) -> None:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if self.last_test_result is None:
            test_status = "not run"
        elif self.last_test_result.passed:
            test_status = "passed"
        else:
            test_status = "failed"
        console.print(
            Panel.fit(
                f"[bold]Furrow Summary[/bold]\n"
                f"Cycles executed: {self.cycles}\n"
                f"Total tasks: {len(tasks)}\n"
                f"Completed: {completed}\n"
                f"Failed: {failed}\n"
                f"Last test result: {test_status}",
                title="Summary",
                border_style="magenta",
            )
        )
