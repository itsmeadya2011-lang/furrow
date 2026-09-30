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
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger("furrow.orchestrator")
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = self.client.settings.max_cycles
        self.max_parallel_tasks = self.client.settings.max_parallel_tasks
        self.tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self._test_result: TestResult | None = None
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_parallel_tasks)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while not self._is_done():
            self.cycles += 1
            logger.info("starting cycle", cycle=self.cycles, goal=self.goal)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
        if self._test_result and self._test_result.passed:
            logger.info("goal complete", cycles=self.cycles)
            console.print("[bold green]Goal complete. Halting.[/bold green]")
        else:
            logger.info("halting", cycles=self.cycles)
            console.print("[yellow]Halting: no more cycles remaining.[/yellow]")

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._test_result = TestResult(
                passed=True, summary="No tasks to complete", failures=[]
            )
            self.cycle_history.append(
                {
                    "cycle": self.cycles,
                    "plan": plan.model_dump(),
                    "task_results": {},
                    "test_result": self._test_result.model_dump(),
                }
            )
            return

        self._semaphore = asyncio.Semaphore(self.max_parallel_tasks)
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self.cycle_history.append(
            {
                "cycle": self.cycles,
                "plan": plan.model_dump(),
                "task_results": {t.id: t.model_dump() for t in plan.tasks},
                "test_result": test_result.model_dump(),
            }
        )

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            worker = WorkerAgent(task=task, client=self.client)
            return await worker.run()

    def _is_done(self) -> bool:
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return True
        if self._test_result is None:
            return False
        if not self._test_result.passed:
            return False
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
