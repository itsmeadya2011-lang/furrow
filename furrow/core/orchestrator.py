from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.settings = settings or Settings()
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.cycle_tasks: list[Any] = []
        self.max_retries = 2
        logger.info("orchestrator_initialized", goal=goal, max_cycles=self.settings.max_cycles)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_start", goal=self.goal)
        while True:
            self.cycles += 1
            
            # Check max cycles
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                logger.warning("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break
                
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_complete", cycles=self.cycles)
                break

    def _resolve_task_order(self, tasks: list[Any]) -> list[list[Any]]:
        """Resolve task execution order based on dependencies. Returns list of batches that can run in parallel."""
        task_map = {task.id: task for task in tasks}
        completed = set()
        batches = []
        
        remaining = set(task.id for task in tasks)
        
        while remaining:
            # Find tasks whose dependencies are all completed
            ready = []
            for task_id in remaining:
                task = task_map[task_id]
                deps = set(task.dependencies)
                if deps.issubset(completed):
                    ready.append(task)
            
            if not ready:
                # Circular dependency or missing dependency - run remaining anyway
                ready = [task_map[tid] for tid in remaining]
            
            batches.append(ready)
            for task in ready:
                completed.add(task.id)
                remaining.remove(task.id)
        
        return batches

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        # Track tasks from this cycle
        self.cycle_tasks = plan.tasks
        self.all_tasks.extend(plan.tasks)

        # Resolve execution order based on dependencies
        batches = self._resolve_task_order(plan.tasks)
        
        for batch_idx, batch in enumerate(batches):
            if len(batch) == 1:
                console.print(f"[cyan]Executing task {batch[0].id}...[/cyan]")
            else:
                console.print(f"[cyan]Executing batch {batch_idx + 1}/{len(batches)} ({len(batch)} tasks in parallel)...[/cyan]")
            
            with Status(f"[bold yellow]Running batch {batch_idx + 1}/{len(batches)}...", console=console):
                tasks = [
                    self._run_task_with_retry(task)
                    for task in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed after retries: {result}[/red]")
                    logger.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
            # Reset goal to original if tests pass
            self.goal = self.original_goal
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            # Set goal to fix failures, but keep original goal
            self.goal = f"Fix failing tests from previous cycle:\n" + "\n".join(test_result.failures) + f"\n\nOriginal goal: {self.original_goal}"

    async def _run_task_with_retry(self, task: Any) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    console.print(f"[yellow]Task {task.id} attempt {attempt + 1} failed, retrying...[/yellow]")
                    logger.warning("task_retry", task_id=task.id, attempt=attempt + 1, error=str(e))
                    await asyncio.sleep(1)
        logger.error("task_failed_all_retries", task_id=task.id, error=str(last_error))
        raise last_error

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0 and pending == 0:
            return False
        if completed >= len(self.all_tasks) and pending == 0:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
