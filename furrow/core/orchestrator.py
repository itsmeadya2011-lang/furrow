from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


def _configure_logging() -> None:
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, stream=sys.stdout, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


_configure_logging()
log = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.output_callback = output_callback

    def _emit(self, *args: Any, **kwargs: Any) -> None:
        if self.output_callback:
            with console.capture() as capture:
                console.print(*args, **kwargs)
            self.output_callback(capture.get())
        else:
            console.print(*args, **kwargs)

    async def run(self) -> None:
        self._emit(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        log.info("run_start", goal=self.goal)
        while True:
            self.cycles += 1
            log.info("cycle_start", cycle=self.cycles)
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                log.info(
                    "max_cycles_reached",
                    cycles=self.cycles,
                    max_cycles=settings.max_cycles,
                )
                self._emit("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                log.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await PlannerAgent(client=self.client).plan(self.goal)
        log.info(
            "plan_received",
            task_count=len(plan.tasks),
            rationale=plan.rationale,
        )
        self.tasks = plan.tasks
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            log.info("no_tasks_planned")
            return

        limit = min(settings.max_parallel_tasks or 1, len(plan.tasks))
        semaphore = asyncio.Semaphore(limit)

        async def _run_task(task: TaskModel) -> TaskModel:
            async with semaphore:
                log.info("task_start", task_id=task.id)
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                except Exception as exc:
                    task.status = "failed"
                    task.result = str(exc)
                    log.error("task_failed", task_id=task.id, error=str(exc))
                    return task
                task.status = "completed"
                task.result = result
                log.info("task_completed", task_id=task.id)
                return task

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            done = await asyncio.gather(*[_run_task(t) for t in plan.tasks])

        for task in done:
            if task.status == "completed":
                self._emit(f"[green]Task {task.id} completed[/green]")
            else:
                self._emit(f"[red]Task {task.id} failed: {task.result}[/red]")

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if self.last_test_result.passed:
            self._emit(f"[green]Tests passed: {self.last_test_result.summary}[/green]")
            log.info("tests_passed", summary=self.last_test_result.summary)
        else:
            self._emit(f"[red]Tests failed: {self.last_test_result.summary}[/red]")
            for failure in self.last_test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self.last_test_result.failures)
            log.warning("tests_failed", failures=self.last_test_result.failures)

    def _is_done(self) -> bool:
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            log.info(
                "max_cycles_reached",
                cycles=self.cycles,
                max_cycles=settings.max_cycles,
            )
            return True

        total = len(self.tasks)
        if total == 0:
            return True

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        tests_passed = self.last_test_result is None or self.last_test_result.passed

        if failed > 0 and not tests_passed:
            return False

        if completed >= total and tests_passed:
            return True

        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
