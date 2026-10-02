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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
_logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.last_test_result: TestResult | None = None
        _logger.info("orchestrator.init", goal=goal, max_cycles=self.client.settings.max_cycles)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            _logger.info("cycle.start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                _logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        _logger.info("plan.created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            _logger.warning("plan.no_tasks")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results, strict=True):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                _logger.error("task.failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                _logger.info("task.completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if self.last_test_result.passed:
            console.print(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
            _logger.info("tests.passed", summary=self.last_test_result.summary)
        else:
            console.print(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            _logger.warning("tests.failed", summary=self.last_test_result.summary)
            self.goal = "Fix failing tests:\n" + "\n".join(self.last_test_result.failures)

    def _is_done(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            _logger.info("max_cycles_reached", cycles=self.cycles, max_cycles=max_cycles)
            console.print(
                f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]"
            )
            return True

        if self.last_test_result is not None and not self.last_test_result.passed:
            _logger.info("not_done_tests_failed")
            return False

        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = any(t.status == "failed" for t in tasks)
        pending = any(t.status == "pending" for t in tasks)
        if failed or pending:
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.plan.tasks if self.plan else []
