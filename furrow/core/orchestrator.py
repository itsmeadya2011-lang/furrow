from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
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

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]

# Delay between consecutive failed cycles, doubling per failure up to the cap.
CYCLE_BACKOFF_BASE_SECONDS = 1.0
CYCLE_BACKOFF_MAX_SECONDS = 30.0


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client if client is not None else LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.test_passed = False
        self.last_test_result: TestResult | None = None
        self.consecutive_failures = 0
        self.stop_event = asyncio.Event()
        self._halt_reason: str | None = None

    async def emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(kind, data)
        except Exception:
            log.exception("event sink failed for %s", kind)

    def stop(self) -> None:
        self.stop_event.set()

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        reason = "goal complete"
        try:
            while True:
                if self.stop_event.is_set():
                    reason = "stopped"
                    break
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self.emit(
                    "cycle_start",
                    {"cycle": self.cycles, "goal": self.original_goal},
                )
                try:
                    await self._cycle()
                except Exception as exc:
                    log.exception("cycle %s failed", self.cycles)
                    self.consecutive_failures += 1
                    await self.emit("error", {"message": str(exc)})
                await self.emit(
                    "cycle_end",
                    {"cycle": self.cycles, "done": self._is_done()},
                )
                if self._is_done():
                    reason = self._halt_reason or "goal complete"
                    break
                if self._max_cycles_reached():
                    reason = "max cycles reached"
                    break
                if self._circuit_breaker_tripped():
                    reason = "max consecutive failures reached"
                    break
                if self.consecutive_failures > 0:
                    await self._sleep_before_retry()
        finally:
            await self.emit("halt", {"reason": reason})
            console.print(f"[bold green]Halting: {reason}[/bold green]")

    async def _sleep_before_retry(self) -> None:
        """Wait before retrying, backing off exponentially on repeated failures.

        The wait races ``stop_event`` so a stop request is never delayed by the
        backoff.
        """
        failures = max(1, self.consecutive_failures)
        exponent = min(failures - 1, 16)
        delay = min(
            CYCLE_BACKOFF_BASE_SECONDS * 2**exponent,
            CYCLE_BACKOFF_MAX_SECONDS,
        )
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.stop_event.wait(), timeout=delay)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        await self.emit("plan", {"cycle": self.cycles, "plan": plan.model_dump()})
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned.[/yellow]")
            self.tasks = []
            satisfied = (
                self.last_test_result is not None
                and self.last_test_result.passed
            )
            self.test_passed = satisfied
            if satisfied:
                self._halt_reason = "goal complete"
            else:
                self.consecutive_failures += 1
            return

        self.tasks = plan.tasks
        semaphore = asyncio.Semaphore(
            max(1, self.client.settings.max_parallel_tasks)
        )

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                await self.emit(
                    "task_start",
                    {"task_id": task.id, "description": task.description},
                )
                return await WorkerAgent(task=task, client=self.client).run()

        results = await asyncio.gather(
            *(_run_task(task) for task in plan.tasks),
            return_exceptions=True,
        )

        cycle_failed = False
        for task, result in zip(plan.tasks, results, strict=True):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = f"Exception: {result}"
                cycle_failed = True
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self.emit(
                    "task_failed", {"task_id": task.id, "error": str(result)}
                )
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self.emit(
                    "task_completed",
                    {"task_id": task.id, "result": str(result)},
                )

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )
        self.last_test_result = test_result
        self.test_passed = test_result.passed
        await self.emit(
            "test_result",
            {
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            },
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            if test_result.failures:
                self.goal = (
                    f"{self.original_goal}\nFix failing tests:\n"
                    + "\n".join(test_result.failures)
                )
            else:
                self.goal = self.original_goal
            cycle_failed = True

        if cycle_failed:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def _max_cycles_reached(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        return max_cycles > 0 and self.cycles >= max_cycles

    def _circuit_breaker_tripped(self) -> bool:
        limit = self.client.settings.max_consecutive_failures
        if limit <= 0:
            return False
        return self.consecutive_failures >= limit

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not self.test_passed:
            return False
        failed = sum(1 for task in tasks if task.status == "failed")
        if failed > 0:
            return False
        if not tasks:
            return True
        completed = sum(1 for task in tasks if task.status == "completed")
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks