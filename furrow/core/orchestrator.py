from __future__ import annotations

import asyncio
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

EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._on_event: EventCallback | None = on_event

    async def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a progress event to any registered callback."""
        log.debug("event", type=event_type, data=data)
        if self._on_event is not None:
            await self._on_event(event_type, data)

    async def run(self) -> None:
        log.info("orchestrator_start", goal=self.goal)
        await self._emit("start", {"goal": self.goal})
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            log.info("cycle_start", cycle=self.cycles)
            await self._emit("cycle_start", {"cycle": self.cycles})
            console.print(
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
            )
            await self._cycle()
            if self._is_done():
                console.print(
                    "[bold green]Goal complete. Halting.[/bold green]"
                )
                log.info("orchestrator_complete", cycles=self.cycles)
                await self._emit(
                    "complete", {"cycles": self.cycles, "goal": self.goal}
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        await self._emit(
            "plan", {"cycle": self.cycles, "plan": plan.model_dump()}
        )

        if not plan.tasks:
            console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        # Store current cycle tasks for state tracking
        self.tasks = plan.tasks

        max_parallel = self.client.settings.max_parallel_tasks
        self._semaphore = asyncio.Semaphore(max_parallel)
        log.info(
            "executing_tasks",
            count=len(plan.tasks),
            max_parallel=max_parallel,
        )

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=console
        ):
            coros = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(
                    f"[red]Task {task.id} failed: {result}[/red]"
                )
                log.error(
                    "task_failed", task_id=task.id, error=str(result)
                )
                await self._emit(
                    "task_failed",
                    {"task_id": task.id, "error": str(result)},
                )
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                log.info("task_completed", task_id=task.id)
                await self._emit(
                    "task_completed",
                    {"task_id": task.id, "result": result[:200]},
                )

        with Status("[bold yellow]Testing...", console=console):
            self._last_test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)

        test_result = self._last_test_result
        if test_result.passed:
            console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
            log.info("tests_passed", summary=test_result.summary)
            await self._emit(
                "tests_passed",
                {"summary": test_result.summary, "failures": []},
            )
        else:
            console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = (
                "Fix failing tests:\n" + "\n".join(test_result.failures)
            )
            log.warning("tests_failed", summary=test_result.summary)
            await self._emit(
                "tests_failed",
                {
                    "summary": test_result.summary,
                    "failures": test_result.failures,
                },
            )

    async def _run_worker(self, task: TaskModel) -> str:
        """Execute a single worker task, bounded by the parallelism semaphore."""
        assert self._semaphore is not None
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        # Respect max_cycles limit (0 means infinite)
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            log.info(
                "max_cycles_reached",
                cycles=self.cycles,
                max_cycles=max_cycles,
            )
            return True

        # If tests failed, continue iterating to fix them
        if (
            self._last_test_result is not None
            and not self._last_test_result.passed
        ):
            return False

        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False

        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        """Return the current cycle's tasks."""
        return self.tasks
