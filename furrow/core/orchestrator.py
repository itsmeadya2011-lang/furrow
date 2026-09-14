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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.state_file = state_file or (settings.workspace / ".furrow_state.json")
        self.task_history: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.task_history = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.task_history))
            except Exception as e:
                logger.warning("failed_to_load_state", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.task_history],
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("state_saved", cycles=self.cycles)
        except Exception as e:
            logger.warning("failed_to_save_state", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Build context from task history
        context = self._build_context()

        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            execute_task = progress.add_task("Executing tasks...", total=len(plan.tasks))

            async def run_worker(task: TaskModel) -> TaskModel:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    task.status = "completed"
                    task.result = result
                    logger.info("task_completed", task_id=task.id)
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    logger.error("task_failed", task_id=task.id, error=str(e))
                progress.advance(execute_task)
                return task

            tasks = await asyncio.gather(*[run_worker(t) for t in plan.tasks], return_exceptions=True)

        # Update task history
        for task in plan.tasks:
            self.task_history.append(task)

        # Display results
        for task in plan.tasks:
            if task.status == "failed":
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")
            else:
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Add test failures to goal for next cycle
            self.goal = f"Fix failing tests from previous cycle:\n" + "\n".join(test_result.failures)

    def _build_context(self) -> str:
        if not self.task_history:
            return ""

        completed = [t for t in self.task_history if t.status == "completed"]
        failed = [t for t in self.task_history if t.status == "failed"]

        parts = ["## Previous Cycles Context"]
        if completed:
            parts.append("\n### Completed Tasks:")
            for t in completed:
                parts.append(f"  - {t.id}: {t.description} ({t.result[:100] if t.result else 'done'})")
        if failed:
            parts.append("\n### Failed Tasks (need retry):")
            for t in failed:
                parts.append(f"  - {t.id}: {t.description} - {t.result}")

        return "\n".join(parts)

    def _is_done(self) -> bool:
        if not self.task_history:
            return False

        # Check if all tasks in the latest plan are completed
        # For simplicity, check if all tasks ever created are completed
        # and no new tasks were planned in the last cycle
        completed = sum(1 for t in self.task_history if t.status == "completed")
        failed = sum(1 for t in self.task_history if t.status == "failed")

        if failed > 0:
            return False

        # If we have tasks and all are completed, we might be done
        # But we need to check if the last cycle produced no new tasks
        return completed >= len(self.task_history) and len(self.task_history) > 0

    def _get_tasks(self) -> list[Any]:
        return self.task_history
