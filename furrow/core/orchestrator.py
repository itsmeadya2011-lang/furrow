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
from furrow.config import Plan, TestResult, TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger
from furrow.state import StateManager

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._all_tasks: list[TaskModel] = []
        self._state_manager = StateManager()
        
        if resume:
            self._resume_from_state()

    def _resume_from_state(self) -> None:
        """Resume from saved state if available."""
        state = self._state_manager.load_state()
        if state:
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            self._all_tasks = self._state_manager.restore_tasks(state)
            logger.info("resumed_from_state", cycles=self.cycles, tasks=len(self._all_tasks))
            console.print(f"[cyan]Resumed from previous session (cycle {self.cycles}, {len(self._all_tasks)} tasks)[/cyan]")

    async def run(self) -> None:
        logger.info("starting_orchestrator", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            
            # Save state after each cycle
            self._state_manager.save_state(self.goal, self._all_tasks, self.cycles)
            
            if self._is_done():
                logger.info("goal_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._state_manager.clear_state()
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        logger.info("plan_created", tasks_count=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            if task.id not in [t.id for t in self._all_tasks]:
                self._all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id, result=result)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Attempting to fix failures...[/yellow]")
            
            # Try to fix failures
            tester = TesterAgent(client=self.client)
            fixes = await tester.fix_failures(self.goal, test_result.failures, plan.tasks)
            
            if fixes:
                for fix in fixes:
                    try:
                        await self.client.write_file(fix["path"], fix["content"])
                        console.print(f"[cyan]Applied fix to {fix['path']}[/cyan]")
                    except Exception as e:
                        logger.error("fix_failed", path=fix.get("path"), error=str(e))
                console.print("[green]Fixes applied. Will re-test in next cycle.[/green]")
            else:
                console.print("[yellow]Could not generate fixes. Will retry in next cycle.[/yellow]")
            
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._all_tasks
