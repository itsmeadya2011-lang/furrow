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
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

console = Console()
configure_logging()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = state_file or (self.config.workspace / ".furrow_state.json")
        self._load_state()
        logger.info("orchestrator_initialized", goal=goal, cycles=self.cycles, existing_tasks=len(self.all_tasks))

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.original_goal = data.get("original_goal", self.original_goal)
                console.print(f"[dim]Loaded state: {self.cycles} cycles, {len(self.all_tasks)} tasks[/dim]")
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.all_tasks))
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "original_goal": self.original_goal,
                "current_goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("state_saved", cycles=self.cycles, tasks=len(self.all_tasks))
        except Exception as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
            title="Furrow"
        ))
        logger.info("run_started", original_goal=self.original_goal)
        while True:
            self.cycles += 1
            
            # Check max cycles
            if self.config.max_cycles > 0 and self.cycles > self.config.max_cycles:
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Stopping.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=self.config.max_cycles)
                break
                
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("run_completed", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        logger.info("cycle_started", cycle=self.cycles, goal=self.goal[:100])
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, config=self.config).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        completed_count = 0
        failed_count = 0
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
                failed_count += 1
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id, result=result[:100] if result else None)
                completed_count += 1

        logger.info("execution_completed", cycle=self.cycles, completed=completed_count, failed=failed_count)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, config=self.config).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            # Reset goal to original if we were fixing tests
            if self.goal != self.original_goal:
                self.goal = self.original_goal
                logger.info("goal_reset_to_original", cycle=self.cycles)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            # Preserve original goal but set current goal to fix tests
            self.goal = f"Original goal: {self.original_goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        
        if failed > 0 and self.goal == self.original_goal:
            # Has failed tasks and not in fix mode
            return False
        if pending == 0 and completed > 0:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
