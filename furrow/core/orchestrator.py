from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import datetime
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
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

configure_logging(settings.log_level)
logger = get_logger(__name__)

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings_obj or settings)
        self.settings = settings_obj or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_tasks: list[TaskModel] = []
        self.failed_tasks: list[TaskModel] = []
        self.state_file = state_file or (self.settings.workspace / ".furrow_state.json")
        self._shutdown = False
        self._setup_signal_handlers()
        self._load_state()

    def _setup_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_shutdown)
            except (ValueError, OSError):
                pass

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        console.print("\n[yellow]Shutdown signal received. Saving state...[/yellow]")
        self._shutdown = True

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("all_tasks", [])]
                self.completed_tasks = [TaskModel(**t) for t in data.get("completed_tasks", [])]
                self.failed_tasks = [TaskModel(**t) for t in data.get("failed_tasks", [])]
                console.print(f"[dim]Loaded state: {self.cycles} cycles, {len(self.all_tasks)} tasks[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "cycles": self.cycles,
                "goal": self.goal,
                "all_tasks": [t.model_dump() for t in self.all_tasks],
                "completed_tasks": [t.model_dump() for t in self.completed_tasks],
                "failed_tasks": [t.model_dump() for t in self.failed_tasks],
                "updated_at": datetime.utcnow().isoformat(),
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")

    async def run(self) -> None:
        logger.info("furrow_started", goal=self.goal, max_cycles=self.settings.max_cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if self.settings.max_cycles > 0:
            console.print(f"[dim]Max cycles: {self.settings.max_cycles}[/dim]")
        while not self._shutdown:
            self.cycles += 1
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]")
                logger.warning("max_cycles_reached", cycles=self.settings.max_cycles)
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_completed", cycles=self.cycles)
                break
        self._save_state()
        logger.info("furrow_stopped", cycles=self.cycles)

    async def _cycle(self) -> None:
        logger.info("planning_started", cycle=self.cycles)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self._get_context())
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("planning_completed", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned", cycle=self.cycles)
            return

        for task in plan.tasks:
            if task.id not in [t.id for t in self.all_tasks]:
                self.all_tasks.append(task)

        logger.info("execution_started", cycle=self.cycles, task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

            async def run_with_semaphore(task: TaskModel) -> str | Exception:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_tasks.append(task)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.completed_tasks.append(task)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        self._print_task_summary()

        logger.info("testing_started", cycle=self.cycles)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_context(self) -> str:
        if not self.all_tasks:
            return ""
        lines = ["## Previous Tasks"]
        for task in self.all_tasks:
            status_icon = "✓" if task.status == "completed" else "✗" if task.status == "failed" else "○"
            lines.append(f"- {status_icon} [{task.id}] {task.description} ({task.status})")
            if task.result:
                lines.append(f"  Result: {task.result[:200]}...")
        return "\n".join(lines)

    def _print_task_summary(self) -> None:
        table = Table(title="Task Summary")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="green")
        table.add_column("Files", style="dim")
        for task in self.all_tasks:
            status_color = {"completed": "green", "failed": "red", "pending": "yellow"}.get(task.status, "white")
            table.add_row(task.id, task.description[:60], f"[{status_color}]{task.status}[/{status_color}]", ", ".join(task.files[:3]))
        console.print(table)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        if failed > 0 and pending == 0:
            return False
        if pending == 0 and completed == len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
