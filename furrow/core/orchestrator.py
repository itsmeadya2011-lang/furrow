from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from structlog import get_logger

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

logger = get_logger()

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_log: Callable[[str], None] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.on_log = on_log
        self.cycles = 0
        self.max_cycles = self.client.settings.max_cycles
        self.max_parallel_tasks = self.client.settings.max_parallel_tasks
        self.current_tasks: list[TaskModel] = []
        self.test_result: TestResult | None = None
        self.failure_context: list[str] = []

    async def run(self) -> None:
        self._emit(
            plain="Furrow\nGoal: " + self.original_goal,
            rich=Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            ),
        )
        logger.info("orchestrator started", goal=self.original_goal)
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self._emit(
                    plain=f"Reached max_cycles ({self.max_cycles}). Stopping.",
                    rich=f"[yellow]Reached max_cycles ({self.max_cycles}). Stopping.[/yellow]",
                )
                logger.info("max_cycles reached, stopping", cycles=self.cycles)
                break

            self.cycles += 1
            self._emit(
                plain=f"Cycle {self.cycles}",
                rich=f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]",
            )
            logger.info("cycle start", cycle=self.cycles)
            await self._cycle()

            if self._is_done():
                self._emit(
                    plain="Goal complete. Halting.",
                    rich="[bold green]Goal complete. Halting.[/bold green]",
                )
                logger.info("goal complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        planning_goal = self.original_goal
        if self.failure_context:
            planning_goal = (
                self.original_goal
                + "\n\nPrevious failures to address:\n"
                + "\n".join(self.failure_context)
            )
        logger.info("planning", goal=planning_goal, cycle=self.cycles)

        with self._status("Planning..."):
            plan: Plan = await self.planner.plan(planning_goal)

        self._emit_plan(plan)
        self.current_tasks = plan.tasks

        if not plan.tasks:
            self._emit(
                plain="No tasks planned. Goal may be complete.",
                rich="[yellow]No tasks planned. Goal may be complete.[/yellow]",
            )
            logger.info("no tasks planned")
            return

        sem = asyncio.Semaphore(self.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with sem:
                logger.info("executing task", task_id=task.id, cycle=self.cycles)
                return await WorkerAgent(task=task, client=self.client).run()

        with self._status("Executing tasks in parallel..."):
            results = await asyncio.gather(
                *(_run_task(t) for t in plan.tasks), return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self._emit(
                    plain=f"Task {task.id} failed: {result}",
                    rich=f"[red]Task {task.id} failed: {result}[/red]",
                )
                logger.error("task failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self._emit(
                    plain=f"Task {task.id} completed",
                    rich=f"[green]Task {task.id} completed[/green]",
                )
                logger.info("task completed", task_id=task.id)

        with self._status("Testing..."):
            try:
                self.test_result = await self.tester.run(self.original_goal, plan.tasks)
            except Exception as e:
                self.test_result = TestResult(passed=False, summary=str(e), failures=[str(e)])
                logger.error("tester error", error=str(e))

        if self.test_result.passed:
            self._emit(
                plain=f"Tests passed: {self.test_result.summary}",
                rich=f"[green]Tests passed: {self.test_result.summary}[/green]",
            )
            logger.info("tests passed", summary=self.test_result.summary)
        else:
            self._emit(
                plain=f"Tests failed: {self.test_result.summary}",
                rich=f"[red]Tests failed: {self.test_result.summary}[/red]",
            )
            for failure in self.test_result.failures:
                self._emit(
                    plain=f"  - {failure}",
                    rich=f"[red]  - {failure}[/red]",
                )
            self._emit(
                plain="Will attempt fix in next cycle.",
                rich="[yellow]Will attempt fix in next cycle.[/yellow]",
            )
            logger.info(
                "tests failed, will retry",
                failures=self.test_result.failures,
                summary=self.test_result.summary,
            )
            self.failure_context = list(self.test_result.failures)
            if self.test_result.summary:
                self.failure_context.append(self.test_result.summary)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        any_failed = any(t.status == "failed" for t in tasks)
        any_pending = any(t.status not in ("completed", "failed") for t in tasks)
        if any_failed or any_pending:
            return False
        return self.test_result is None or self.test_result.passed

    def _get_tasks(self) -> list[Any]:
        return self.current_tasks

    def _status(self, text: str) -> Any:
        if self.on_log is not None:
            self.on_log(text)
            return nullcontext()
        return Status(text, console=console)

    def _emit(self, plain: str, rich: Any = None) -> None:
        if self.on_log is not None:
            self.on_log(plain)
        else:
            console.print(rich if rich is not None else plain)

    def _emit_plan(self, plan: Plan) -> None:
        dumped = json.dumps(plan.model_dump(), indent=2, default=str)
        self._emit(
            plain="Plan: " + dumped,
            rich=Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"),
        )
