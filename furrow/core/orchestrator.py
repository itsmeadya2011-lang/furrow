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
from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.original_goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        logger.info("Starting Furrow orchestrator", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.warning("Max cycles reached", max_cycles=self.settings.max_cycles)
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            
            self.cycles += 1
            logger.info("Starting cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("Goal satisfied, halting")
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Build context from previous cycles
        context = self._build_context()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context)
        logger.info("Plan created", cycle=self.cycles, task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("No tasks planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks
        self.all_tasks.extend(plan.tasks)
        
# Limit parallel execution
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        
        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                try:
                    result = await worker.run()
                    return result
                except Exception as e:
                    logger.error("Task raised exception", task_id=task.id, error=str(e))
                    raise

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_results: list[dict[str, Any]] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("Task failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                cycle_results.append({"task_id": task.id, "status": "failed", "result": str(result)})
            else:
                task.status = "completed"
                task.result = result
                logger.info("Task completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                cycle_results.append({"task_id": task.id, "status": "completed", "result": result})

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self.all_tasks)

        cycle_summary = {
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": cycle_results,
            "test_result": test_result.model_dump(),
        }
        self.cycle_history.append(cycle_summary)

        if test_result.passed:
            logger.info("Tests passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            # Check if original goal is satisfied
            if self._is_goal_satisfied():
                return
        else:
            logger.warning("Tests failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests from previous cycle:\n" + "\n".join(test_result.failures)

    def _build_context(self) -> str:
        """Build context from previous cycles for the planner."""
        if not self.cycle_history:
            return ""
        
        parts = ["Previous cycles summary:"]
        for cycle in self.cycle_history[-3:]:  # Last 3 cycles
            parts.append(f"\nCycle {cycle['cycle']}:")
            parts.append(f"  Goal: {cycle['goal']}")
            for task in cycle['tasks']:
                parts.append(f"  - Task {task['task_id']}: {task['status']}")
                if task['result']:
                    parts.append(f"    Result: {task['result'][:200]}")
            tr = cycle['test_result']
            parts.append(f"  Tests: {'PASSED' if tr['passed'] else 'FAILED'} - {tr['summary']}")
            if tr['failures']:
                for f in tr['failures']:
                    parts.append(f"    Failure: {f}")
        return "\n".join(parts)

    def _is_goal_satisfied(self) -> bool:
        """Check if the original goal has been satisfied."""
        # If all tasks from all cycles are completed and tests pass, we're done
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        return failed == 0 and completed == len(self.all_tasks)

    def _is_done(self) -> bool:
        return self._is_goal_satisfied()

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
