from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import structlog
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


def configure_logger(log_format: str, log_level: str) -> structlog.BoundLogger:
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    processors: list[Any] = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
    )
    return structlog.get_logger("furrow")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_message: Callable[[str], None] | None = None,
    ) -> None:
        self.original_goal: str = goal
        self.goal: str = goal
        self.client: LLMClient = client or LLMClient()
        self.settings: Settings = self.client.settings
        self.planner: PlannerAgent = PlannerAgent(client=self.client)
        self.cycles: int = 0
        self.all_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._current_tasks: list[TaskModel] = []
        self._semaphore: asyncio.Semaphore | None = None
        self.logger = configure_logger(self.settings.log_format, self.settings.log_level)
        self.on_message: Callable[[str], None] | None = on_message

    def _emit(self, message: str) -> None:
        """Send a progress message to the on_message callback if registered."""
        if self.on_message is not None:
            try:
                self.on_message(message)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("on_message_error", error=str(exc))

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        return self._semaphore

    async def run(self) -> None:
        self.logger.info("furrow_start", goal=self.original_goal)
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            self.logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(f"[Cycle {self.cycles}] Starting planning...")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self.logger.info("goal_complete", cycles=self.cycles)
                self._emit("[Furrow] Goal complete. Halting.")
                break

    def _build_context(self) -> str:
        parts: list[str] = [f"Original Goal: {self.original_goal}"]
        completed = [t for t in self.all_tasks if t.status == "completed"]
        failed = [t for t in self.all_tasks if t.status == "failed"]
        if completed:
            parts.append("Completed tasks in previous cycles:")
            for t in completed:
                detail = f"  - [{t.id}] {t.description}"
                if t.result:
                    detail += f" :: {t.result}"
                parts.append(detail)
        if failed:
            parts.append("Previously failed tasks:")
            for t in failed:
                detail = f"  - [{t.id}] {t.description}"
                if t.result:
                    detail += f" :: {t.result}"
                parts.append(detail)
        if self.last_test_result is not None:
            if self.last_test_result.passed:
                parts.append(
                    f"Last test result: PASSED ({self.last_test_result.summary})"
                )
                parts.append(
                    "Objective: All tests pass; continue only if the goal is unfinished."
                )
            else:
                parts.append("Last test result: FAILED")
                for failure in self.last_test_result.failures:
                    parts.append(f"  - {failure}")
                parts.append(
                    "Objective: Fix the failing tests while continuing toward the original goal."
                )
        else:
            parts.append("Objective: Complete the original goal.")
        return "\n".join(parts)

    async def _cycle(self) -> TestResult:
        context = self._build_context()
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(context)
        self.logger.info(
            "plan_created", cycle=self.cycles, tasks=len(plan.tasks), rationale=plan.rationale
        )
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_result = TestResult(passed=True, summary="No tasks planned")
            self._current_tasks = []
            return self.last_test_result

        self._current_tasks = list(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[self._run_task(task) for task in plan.tasks],
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self.logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit(f"[Cycle {self.cycles}] Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self.logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                self._emit(f"[Cycle {self.cycles}] Task {task.id} completed")

        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.original_goal, plan.tasks
            )
        self.last_test_result = test_result
        self.logger.info(
            "test_result",
            cycle=self.cycles,
            passed=test_result.passed,
            summary=test_result.summary,
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"[Cycle {self.cycles}] Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._emit(
                f"[Cycle {self.cycles}] Tests failed. Will attempt fix: {test_result.summary}"
            )
            self.goal = f"Fix failing tests: {', '.join(test_result.failures)}"

        return test_result

    async def _run_task(self, task: TaskModel) -> str:
        async with self.semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        max_cycles: int = self.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            self.logger.info(
                "max_cycles_reached", cycles=self.cycles, max_cycles=max_cycles
            )
            self._emit(
                f"[Furrow] Max cycles ({max_cycles}) reached. Halting."
            )
            return True
        if not self._current_tasks:
            return True
        all_completed = all(t.status == "completed" for t in self._current_tasks)
        if (
            self.last_test_result is not None
            and self.last_test_result.passed
            and all_completed
        ):
            self.logger.info("goal_complete", cycles=self.cycles)
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
