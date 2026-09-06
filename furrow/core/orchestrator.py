from __future__ import annotations

import asyncio
from pathlib import Path

import structlog

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger("furrow")

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("starting_cycle", cycle=self.cycles, goal=self.goal)
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
        except Exception as e:
            logger.error("planning_failed", error=str(e), goal=self.goal)
            console.print(f"[red]Planning failed: {e}[/red]")
            self.goal = f"Fix planning error:\n{str(e)}"
            return
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.current_tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
        else:
            logger.error("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.info("goal_updated_for_fix", goal=self.goal)

    def _is_done(self) -> bool:
        if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
            logger.info(
                "max_cycles_reached",
                cycles=self.cycles,
                max_cycles=self.client.settings.max_cycles,
            )
            return True
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.current_tasks
