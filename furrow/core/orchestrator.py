from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, Settings, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()
STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.planner = PlannerAgent(client=self.client, settings=settings)
        self.settings = settings or Settings()
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []  # Track all tasks across cycles
        self.cycle_plans: list[Plan] = []  # Track plans for context
        self.state_file = self.settings.workspace / STATE_FILE
        self.max_retries = 3
        
        if resume:
            self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("all_tasks", [])]
                self.cycle_plans = [Plan(**p) for p in data.get("cycle_plans", [])]
                console.print(f"[green]Resumed from state: {len(self.all_tasks)} tasks, {self.cycles} cycles[/green]")
                logger.info("state_loaded", tasks=len(self.all_tasks), cycles=self.cycles)
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")
                logger.error("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "all_tasks": [t.model_dump() for t in self.all_tasks],
                "cycle_plans": [p.model_dump() for p in self.cycle_plans],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("state_saved", tasks=len(self.all_tasks), cycles=self.cycles)
        except Exception as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")
            logger.error("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)
        while True:
            # Check max_cycles limit (0 = infinite)
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break
            
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", cycles=self.cycles)
                # Clean up state file on completion
                if self.state_file.exists():
                    self.state_file.unlink()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.cycle_plans, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        self.cycle_plans.append(plan)
        
        # Add new tasks to tracking
        for task in plan.tasks:
            self.all_tasks.append(task)

        # Display task table
        self._display_tasks(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results and retry failed tasks
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        # Retry failed tasks
        await self._retry_failed_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _retry_failed_tasks(self, tasks: list[TaskModel]) -> None:
        """Retry failed tasks up to max_retries times."""
        failed_tasks = [t for t in tasks if t.status == "failed"]
        if not failed_tasks:
            return
        
        console.print(f"[yellow]Retrying {len(failed_tasks)} failed task(s)...[/yellow]")
        for task in failed_tasks:
            retry_count = getattr(task, 'retry_count', 0)
            if retry_count >= self.max_retries:
                console.print(f"[red]Task {task.id} exceeded max retries ({self.max_retries})[/red]")
                logger.error("task_max_retries_exceeded", task_id=task.id, max_retries=self.max_retries)
                continue
            
            task.retry_count = retry_count + 1
            task.status = "pending"
            console.print(f"[cyan]Retrying task {task.id} (attempt {task.retry_count}/{self.max_retries})[/cyan]")
            logger.info("task_retry", task_id=task.id, attempt=task.retry_count)
            
            try:
                result = await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed on retry[/green]")
                logger.info("task_retry_succeeded", task_id=task.id)
            except Exception as e:
                task.status = "failed"
                task.result = str(e)
                console.print(f"[red]Task {task.id} failed on retry: {e}[/red]")
                logger.error("task_retry_failed", task_id=task.id, error=str(e))

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0:
            return False
        if pending > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks

    def _display_tasks(self, tasks: list[Any]) -> None:
        table = Table(title="Tasks This Cycle")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Files", style="yellow")
        table.add_column("Dependencies", style="magenta")
        for task in tasks:
            table.add_row(
                task.id,
                task.description[:60] + "..." if len(task.description) > 60 else task.description,
                ", ".join(task.files) if task.files else "any",
                ", ".join(task.dependencies) if task.dependencies else "none"
            )
        console.print(table)
