from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings as default_settings
from furrow.llm import LLMClient

_default_console = Console()
console = _default_console


logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.console = console or _default_console
        self.all_tasks: list[TaskModel] = []
        self.test_results: list[TestResult] = []

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                break
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.all_tasks = plan.tasks

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.test_results.append(test_result)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        if not self.all_tasks:
            return True
        if self.test_results and not self.test_results[-1].passed:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks