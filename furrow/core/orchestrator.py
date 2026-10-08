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

console = Console()
log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._worker_max_retries: int = 3

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator_started", goal=self.goal)
        while True:
            self.cycles += 1
            log.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            if self.client.settings.max_cycles > 0 and self.cycles > self.client.settings.max_cycles:
                log.warning("max_cycles_reached", max_cycles=self.client.settings.max_cycles, current_cycle=self.cycles)
                console.print(f"[yellow]Max cycles ({self.client.settings.max_cycles}) reached. Halting.[/yellow]")
                break

            await self._cycle()
            if self._is_done():
                log.info("orchestrator_done", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._tasks = plan.tasks
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [self._run_worker_with_retry(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
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

        self._last_test_result = test_result

        if test_result.passed:
            log.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker_with_retry(self, task: TaskModel) -> str:
        last_exception: Exception | None = None
        for attempt in range(self._worker_max_retries + 1):
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
                if attempt > 0:
                    log.info("worker_retry_succeeded", task_id=task.id, attempt=attempt)
                return result
            except Exception as e:  # noqa: BLE001
                last_exception = e
                if attempt < self._worker_max_retries:
                    log.warning("worker_retry", task_id=task.id, attempt=attempt + 1, max_retries=self._worker_max_retries, error=str(e))
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    log.error("worker_failed_all_retries", task_id=task.id, attempts=attempt + 1, error=str(e))
        raise last_exception

    def _is_done(self) -> bool:
        if self._last_test_result is not None and not self._last_test_result.passed:
            return False
        if not self._tasks:
            return True
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self._tasks if t.status == "completed")
        return completed >= len(self._tasks)

    def _get_tasks(self) -> list[Any]:
        return self._tasks