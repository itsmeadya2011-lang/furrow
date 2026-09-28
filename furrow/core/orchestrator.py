from __future__ import annotations

import asyncio

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger("orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._plan: Plan | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self._max_cycles_reached():
                console.print(f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Stopping.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            await asyncio.sleep(0.1)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        log.info("plan_created", num_tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                log.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                log.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)

    def _max_cycles_reached(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        return max_cycles > 0 and self.cycles >= max_cycles

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        if self._plan is None:
            return []
        return self._plan.tasks
