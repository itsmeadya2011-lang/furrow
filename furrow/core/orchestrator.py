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
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("furrow.orchestrator")

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None,
        progress_callback: ProgressCallback | None = None
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.max_retries = 3
        self.progress_callback = progress_callback

    async def _emit_progress(self, event: dict[str, Any]) -> None:
        """Emit progress event to callback if registered."""
        if self.progress_callback:
            await self.progress_callback(event)

    async def run(self) -> None:
        logger.info("starting_orchestrator", goal=self.goal)
        await self._emit_progress({"type": "start", "goal": self.goal})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check max_cycles limit
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                logger.warning("max_cycles_reached", max_cycles=settings.max_cycles)
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                await self._emit_progress({"type": "max_cycles_reached", "max_cycles": settings.max_cycles})
                break
            
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            await self._emit_progress({"type": "cycle_start", "cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("orchestrator_complete", cycles=self.cycles)
                await self._emit_progress({"type": "complete", "cycles": self.cycles})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Build context from previous cycles
        context = self._build_context()
        
        await self._emit_progress({"type": "planning_start", "cycle": self.cycles})
        with Status("[bold yellow]Planning...", console=console) as status:
            logger.debug("planning_start", cycle=self.cycles)
            plan = await self.planner.plan(self.goal, context)
            logger.debug("planning_complete", cycle=self.cycles, task_count=len(plan.tasks))
        await self._emit_progress({
            "type": "planning_complete", 
            "cycle": self.cycles, 
            "plan": plan.model_dump()
        })
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no_tasks_planned", cycle=self.cycles)
            await self._emit_progress({"type": "no_tasks", "cycle": self.cycles})
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks for this cycle
        cycle_tasks = plan.tasks
        self.all_tasks.extend(cycle_tasks)
        
        # Execute tasks with retry logic
        cycle_result = {
            "cycle": self.cycles,
            "tasks": [],
            "test_result": None,
        }
        
        for task in cycle_tasks:
            await self._execute_task_with_retry(task, cycle_result)

        await self._emit_progress({"type": "testing_start", "cycle": self.cycles})
        with Status("[bold yellow]Testing...", console=console) as status:
            logger.debug("testing_start", cycle=self.cycles)
            test_result = await TesterAgent(client=self.client).run(self.goal, cycle_tasks)
            logger.debug("testing_complete", cycle=self.cycles, passed=test_result.passed)

        cycle_result["test_result"] = {
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures,
        }
        self.cycle_history.append(cycle_result)

        if test_result.passed:
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            await self._emit_progress({
                "type": "tests_passed", 
                "cycle": self.cycles, 
                "summary": test_result.summary
            })
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            await self._emit_progress({
                "type": "tests_failed", 
                "cycle": self.cycles, 
                "summary": test_result.summary,
                "failures": test_result.failures
            })
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_task_with_retry(self, task: Any, cycle_result: dict) -> None:
        """Execute a task with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            if attempt > 0:
                logger.warning("task_retry", task_id=task.id, attempt=attempt + 1, max_retries=self.max_retries)
                console.print(f"[yellow]Retrying task {task.id} (attempt {attempt + 1}/{self.max_retries})[/yellow]")
                await self._emit_progress({
                    "type": "task_retry",
                    "task_id": task.id,
                    "attempt": attempt + 1,
                    "max_retries": self.max_retries
                })
            
            await self._emit_progress({"type": "task_start", "task_id": task.id, "attempt": attempt + 1})
            with Status(f"[bold yellow]Executing task {task.id}...", console=console):
                try:
                    logger.debug("task_execute_start", task_id=task.id, attempt=attempt + 1)
                    result = await WorkerAgent(task=task, client=self.client).run()
                    logger.debug("task_execute_complete", task_id=task.id, result_length=len(result))
                except Exception as e:
                    last_error = e
                    logger.error("task_execute_error", task_id=task.id, error=str(e))
                    await self._emit_progress({
                        "type": "task_error",
                        "task_id": task.id,
                        "error": str(e)
                    })
                    continue
            
            # Success
            task.status = "completed"
            task.result = result
            logger.info("task_completed", task_id=task.id)
            await self._emit_progress({
                "type": "task_complete",
                "task_id": task.id,
                "result": result
            })
            console.print(f"[green]Task {task.id} completed[/green]")
            cycle_result["tasks"].append({"id": task.id, "status": "completed", "result": result})
            return
        
        # All retries failed
        task.status = "failed"
        task.result = str(last_error)
        logger.error("task_failed", task_id=task.id, error=str(last_error), max_retries=self.max_retries)
        await self._emit_progress({
            "type": "task_failed",
            "task_id": task.id,
            "error": str(last_error)
        })
        console.print(f"[red]Task {task.id} failed after {self.max_retries} attempts: {last_error}[/red]")
        cycle_result["tasks"].append({"id": task.id, "status": "failed", "error": str(last_error)})

    def _build_context(self) -> str:
        """Build context string from previous cycles for the planner."""
        if not self.cycle_history:
            return ""
        
        context_parts = ["Previous cycles summary:"]
        for cycle in self.cycle_history:
            context_parts.append(f"\nCycle {cycle['cycle']}:")
            for task in cycle["tasks"]:
                context_parts.append(f"  Task {task['id']}: {task['status']} - {task.get('result', task.get('error', ''))[:100]}")
            tr = cycle["test_result"]
            context_parts.append(f"  Tests: {'passed' if tr['passed'] else 'failed'} - {tr['summary']}")
            if tr["failures"]:
                for f in tr["failures"]:
                    context_parts.append(f"    Failure: {f}")
        return "\n".join(context_parts)

    def _is_done(self) -> bool:
        # Check if all tasks from the latest plan are completed
        if not self.all_tasks:
            return False
            
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        
        # If any task failed, we're not done (will retry in next cycle)
        if failed > 0:
            return False
            
        # Check if we have a successful test run
        if self.cycle_history and self.cycle_history[-1]["test_result"]["passed"]:
            return True
            
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
