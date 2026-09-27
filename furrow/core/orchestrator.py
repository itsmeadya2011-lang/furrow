from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, get_logger
from furrow.llm import LLMClient

console = Console()
logger = get_logger()

_TAG_RE = re.compile(r"\[/?[^\[\]]*\]")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        status_emitter: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.current_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.status_emitter = status_emitter

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._emit(f"Furrow — Goal: {self.goal}")
        logger.info("run_started", goal=self.goal)
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("run_complete", cycles=self.cycles)
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                await self._emit("[bold yellow]Max cycles reached. Halting.[/bold yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles)
                break

    async def _emit(self, message: str) -> None:
        if self.status_emitter is not None:
            await self.status_emitter(_TAG_RE.sub("", message))
        else:
            console.print(message)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {plan.rationale} ({len(plan.tasks)} task(s))")
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.current_tasks = []
            return

        self.current_tasks = plan.tasks
        self.all_tasks.extend(plan.tasks)

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        waves = self._task_waves(plan.tasks)
        results: list[tuple[TaskModel, Any]] = []
        for wave in waves:
            wave_results = await asyncio.gather(
                *[_run(task) for task in wave], return_exceptions=True
            )
            results.extend(zip(wave, wave_results))

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self.last_test_result = test_result

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _task_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        resolved: set[str] = set()
        remaining = list(tasks)
        waves: list[list[TaskModel]] = []
        while remaining:
            wave = [t for t in remaining if all(dep in resolved for dep in t.dependencies)]
            if not wave:
                wave = list(remaining)
            waves.append(wave)
            for task in wave:
                resolved.add(task.id)
                remaining.remove(task)
        return waves

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if self.last_test_result is not None and not self.last_test_result.passed:
            return False
        if any(t.status == "failed" for t in tasks):
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.current_tasks
