from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.config import settings
from furrow.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("orchestrator")
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        workspace: Path | None = None,
        state_file: str | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.workspace = workspace or settings.workspace
        self.state_file = Path(state_file) if state_file else self.workspace / ".furrow_state.json"
        self.all_tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("state_loaded", cycles=self.cycles, task_count=len(self.all_tasks))
                console.print(f"[dim]Loaded state: {self.cycles} cycles, {len(self.all_tasks)} tasks[/dim]")
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "updated": time.time(),
        }
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug("state_saved", cycles=self.cycles, task_count=len(self.all_tasks))

    async def run(self) -> None:
        logger.info("orchestrator_started", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("orchestrator_completed", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("planning_started", cycle=self.cycles, goal=self.goal)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("planning_completed", cycle=self.cycles, task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no_tasks_planned", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks from this cycle
        cycle_tasks = plan.tasks
        self.all_tasks.extend(cycle_tasks)

        logger.info("execution_started", cycle=self.cycles, task_count=len(cycle_tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in cycle_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        completed_count = 0
        failed_count = 0
        for task, result in zip(cycle_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                failed_count += 1
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                completed_count += 1
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        logger.info("execution_completed", cycle=self.cycles, completed=completed_count, failed=failed_count)

        # Print task summary table
        self._print_task_summary()

        logger.info("testing_started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, cycle_tasks)

        if test_result.passed:
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _print_task_summary(self) -> None:
        table = Table(title="Task Summary")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Files", style="dim")
        table.add_column("Status", style="bold")
        table.add_column("Result", style="dim")

        for task in self.all_tasks:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "failed": "red",
            }.get(task.status, "white")
            status_str = f"[{status_color}]{task.status}[/{status_color}]"
            result_preview = (task.result[:50] + "...") if task.result and len(task.result) > 50 else (task.result or "")
            table.add_row(
                task.id,
                task.description,
                ", ".join(task.files) if task.files else "any",
                status_str,
                result_preview,
            )
        console.print(table)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        # Done if no pending tasks and no failed tasks
        if pending == 0 and failed == 0:
            return True
        # Or if max cycles reached
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached.[/yellow]")
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
