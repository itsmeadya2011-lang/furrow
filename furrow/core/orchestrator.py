from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_plan: Plan | None = None
        self.settings = settings or self.client.settings
        self.on_event = on_event

    async def _emit(self, message: str) -> None:
        if self.on_event:
            await self.on_event(message)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator_started", goal=self.goal, max_cycles=self.settings.max_cycles)
        await self._emit(f"START|Goal: {self.goal}")
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._emit(f"CYCLE|Cycle {self.cycles}")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("orchestrator_completed", cycles=self.cycles)
                await self._emit("DONE|Goal complete")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles={self.settings.max_cycles}. Halting.[/yellow]")
                log.info("orchestrator_max_cycles_reached", cycles=self.cycles, max_cycles=self.settings.max_cycles)
                await self._emit("DONE|Max cycles reached")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)
        await self._emit(f"PLAN|{json.dumps(plan.model_dump())}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("NO_TASKS|No tasks planned")
            return

        # Enforce max_parallel_tasks limit
        max_parallel = self.settings.max_parallel_tasks
        task_chunks = [plan.tasks[i:i + max_parallel] for i in range(0, len(plan.tasks), max_parallel)]

        all_results = []
        for chunk in task_chunks:
            with Status(f"[bold yellow]Executing {len(chunk)} tasks in parallel...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in chunk
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(zip(chunk, results))

        for task, result in all_results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("task_failed", task_id=task.id, error=str(result))
                await self._emit(f"TASK_FAILED|{task.id}|{result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                log.info("task_completed", task_id=task.id, result_preview=result[:200])
                await self._emit(f"TASK_DONE|{task.id}|{result[:200]}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", summary=test_result.summary)
            await self._emit(f"TEST_PASS|{test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            await self._emit(f"TEST_FAIL|{test_result.summary}")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
