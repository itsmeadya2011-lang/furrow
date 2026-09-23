from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable, Awaitable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = self.client.settings.max_cycles
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.on_output = on_output
        logger.info("orchestrator_initialized", goal=goal, max_cycles=self.max_cycles)

    async def run(self) -> None:
        await self._emit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}")
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", cycles=self.cycles)
                break
            if self.max_cycles and self.cycles >= self.max_cycles:
                await self._emit(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                logger.warning("max_cycles_reached", cycles=self.cycles, max_cycles=self.max_cycles)
                break

    async def _emit(self, message: str) -> None:
        console.print(message)
        if self.on_output:
            await self.on_output(message)

    async def _cycle(self) -> None:
        # Build context from previous cycles
        context = self._build_context()

        logger.info("planning_started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context)
        logger.info("planning_completed", cycle=self.cycles, task_count=len(plan.tasks))
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our tracked list
        for task in plan.tasks:
            # Check if task already exists (by id)
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if not existing:
                self.all_tasks.append(task)

        logger.info("execution_started", cycle=self.cycles, task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_results = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
                cycle_results.append({"task_id": task.id, "status": "failed", "result": str(result)})
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", cycle=self.cycles, task_id=task.id)
                cycle_results.append({"task_id": task.id, "status": "completed", "result": result})

        logger.info("testing_started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "plan": plan.model_dump(),
            "results": cycle_results,
            "test_result": test_result.model_dump(),
        })

    def _build_context(self) -> str:
        """Build context string from previous cycles for the planner."""
        if not self.cycle_history:
            return "No previous cycles."

        lines = ["Previous cycles:"]
        for cycle in self.cycle_history[-3:]:  # Last 3 cycles
            lines.append(f"\nCycle {cycle['cycle']}:")
            lines.append(f"  Plan: {cycle['plan']['rationale']}")
            for task in cycle['plan']['tasks']:
                lines.append(f"  Task {task['id']}: {task['description']} ({task.get('status', 'pending')})")
            test = cycle['test_result']
            lines.append(f"  Tests: {'PASSED' if test['passed'] else 'FAILED'} - {test['summary']}")
            if test['failures']:
                for f in test['failures']:
                    lines.append(f"    Failure: {f}")
        return "\n".join(lines)

    def _is_done(self) -> bool:
        # Check if all tracked tasks are completed
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def get_state(self) -> dict[str, Any]:
        """Get current state for persistence/inspection."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "history": self.cycle_history,
        }