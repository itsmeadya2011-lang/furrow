from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

logger = structlog.get_logger()
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.state_file = state_file or (self.settings.workspace / ".furrow_state.json")
        self.tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.completed_task_ids = set(data.get("completed_task_ids", []))
                self.failed_task_ids = set(data.get("failed_task_ids", []))
                self.original_goal = data.get("original_goal", self.goal)
                logger.info("Loaded state", cycles=self.cycles, tasks=len(self.tasks))
            except Exception as e:
                logger.warning("Failed to load state", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
                "completed_task_ids": list(self.completed_task_ids),
                "failed_task_ids": list(self.failed_task_ids),
                "original_goal": self.original_goal,
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("Saved state", cycles=self.cycles)
        except Exception as e:
            logger.warning("Failed to save state", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        max_cycles = self.settings.max_cycles
        if max_cycles > 0:
            console.print(f"[dim]Max cycles: {max_cycles}[/dim]")

        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            await self._cycle()
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Filter out already completed tasks
        pending_tasks = [t for t in self.tasks if t.id not in self.completed_task_ids and t.id not in self.failed_task_ids]
        
        # If no pending tasks, plan new ones
        if not pending_tasks:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return

            self.tasks = plan.tasks
            pending_tasks = self.tasks

        # Execute pending tasks in parallel (respecting max_parallel_tasks)
        max_parallel = self.settings.max_parallel_tasks
        for i in range(0, len(pending_tasks), max_parallel):
            batch = pending_tasks[i:i + max_parallel]
            console.print(f"[dim]Executing batch {i//max_parallel + 1}: {len(batch)} tasks[/dim]")
            
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    for task in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self.failed_task_ids.add(task.id)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("Task failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    self.completed_task_ids.add(task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("Task completed", task_id=task.id)

        # Run tests
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Keep original goal but prepend fix context
            self.goal = f"{self.original_goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        
        completed = sum(1 for t in self.tasks if t.id in self.completed_task_ids)
        failed = sum(1 for t in self.tasks if t.id in self.failed_task_ids)
        
        # If any failed, not done (will retry)
        if failed > 0:
            return False
        
        # All tasks completed
        if completed >= len(self.tasks):
            return True
        
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks