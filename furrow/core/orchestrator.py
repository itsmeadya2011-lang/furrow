from __future__ import annotations

import asyncio
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, settings
from furrow.core.state import StateStore
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        max_parallel_tasks: int = 5,
        planner_model: str | None = None,
        worker_model: str | None = None,
        tester_model: str | None = None,
        event_bus: Any | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        self.goal = goal
        self.cycles = 0
        self.max_cycles = max_cycles
        self.max_parallel_tasks = max_parallel_tasks
        self.event_bus = event_bus
        self.state_store = state_store

        overrides: dict[str, str] = {}
        if planner_model is not None:
            overrides["planner_model"] = planner_model
        if worker_model is not None:
            overrides["worker_model"] = worker_model
        if tester_model is not None:
            overrides["tester_model"] = tester_model

        effective_settings = settings.model_copy(update=overrides) if overrides else settings
        self.client = client or LLMClient(settings=effective_settings)
        self.planner = PlannerAgent(client=self.client)
        self.current_plan: Plan | None = None

    async def run(self) -> None:
        """Execute the planner/worker/tester loop until completion or max cycles.

        ``max_cycles`` caps the number of full cycles executed (0 = unlimited).
        """
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                if self.max_cycles > 0 and self.cycles > self.max_cycles:
                    console.print(
                        f"[bold yellow]Max cycles reached ({self.max_cycles}). Halting.[/bold yellow]"
                    )
                    if self.event_bus is not None:
                        self.event_bus.emit_done("max_cycles_reached")
                    break
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                if self.event_bus is not None:
                    self.event_bus.emit_cycle(self.cycles)
                await self._cycle()
                if not self._get_tasks():
                    console.print("[bold yellow]No more tasks. Halting.[/bold yellow]")
                    if self.event_bus is not None:
                        self.event_bus.emit_done("no_more_tasks")
                    break
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    if self.event_bus is not None:
                        self.event_bus.emit_done("goal_complete")
                    break
        finally:
            await self.client.aclose()

    async def _cycle(self) -> None:
        """Run one planner -> worker -> tester cycle."""
        logger.info("planning", extra={"goal": self.goal})
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan ready", extra={"num_tasks": len(plan.tasks)})
        if self.event_bus is not None:
            self.event_bus.emit_plan(plan)
        if self.state_store is not None:
            self.state_store.save_plan(self.goal, self.cycles, plan)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        logger.info("executing tasks", extra={"concurrency": self.max_parallel_tasks})
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.max_parallel_tasks)

            async def _run_limited(task_coro):
                async with semaphore:
                    return await task_coro

            tasks = [_run_limited(WorkerAgent(task=task, client=self.client).run()) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            if self.event_bus is not None:
                self.event_bus.emit_task(task.id, task.status, task.result)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        if self.event_bus is not None:
            self.event_bus.emit_tests(test_result)
        if self.state_store is not None:
            self.state_store.save_cycle_result(
                self.goal, self.cycles, test_result.passed, test_result.summary
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        """Return the tasks from the most recent plan, or [] if no plan yet."""
        if self.current_plan is None:
            return []
        return self.current_plan.tasks
