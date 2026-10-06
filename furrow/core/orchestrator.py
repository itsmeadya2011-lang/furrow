from __future__ import annotations

import asyncio
import json
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
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings = settings,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=config)
        self.planner = PlannerAgent(client=self.client)
        self.config = config
        self.state_file = state_file or config.workspace / ".furrow_state.json"
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("State loaded", cycles=self.cycles, tasks=len(self.all_tasks))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load state", error=str(e))

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.all_tasks],
        }
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug("State saved", file=str(self.state_file))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("Starting orchestrator", goal=self.goal)

        while True:
            if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Halting.[/yellow]")
                logger.info("Max cycles reached", max_cycles=self.config.max_cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("Goal complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        # Plan new tasks or continue with pending ones
        pending_tasks = [t for t in self.all_tasks if t.status == "pending"]
        failed_tasks = [t for t in self.all_tasks if t.status == "failed"]

        if pending_tasks or failed_tasks:
            # Continue with existing tasks
            tasks_to_run = self._get_runnable_tasks(pending_tasks + failed_tasks)
            if tasks_to_run:
                console.print(f"[yellow]Continuing with {len(tasks_to_run)} runnable tasks[/yellow]")
                await self._execute_tasks(tasks_to_run)
                return

        # No pending/failed tasks, plan new ones
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("No tasks planned, goal may be complete")
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        # Execute tasks respecting dependencies
        runnable = self._get_runnable_tasks(plan.tasks)
        if runnable:
            await self._execute_tasks(runnable)

        # Run tests
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("Tests failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_runnable_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks that have no unmet dependencies."""
        completed_ids = {t.id for t in self.all_tasks if t.status == "completed"}
        runnable = []
        for task in tasks:
            if task.status in ("completed", "running"):
                continue
            unmet = [dep for dep in task.dependencies if dep not in completed_ids]
            if not unmet:
                runnable.append(task)
            else:
                logger.debug("Task waiting on dependencies", task=task.id, unmet=unmet)
        return runnable

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        if not tasks:
            return

        # Limit parallel execution
        max_parallel = min(self.config.max_parallel_tasks, len(tasks))
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
            async with semaphore:
                task.status = "running"
                worker = WorkerAgent(task=task, client=self.client)
                try:
                    result = await worker.run()
                    return task, result
                except Exception as e:
                    return task, e

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in tasks],
                return_exceptions=True,
            )

        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("Task failed", task=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("Task completed", task=task.id)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status in ("pending", "running"))

        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed >= len(self.all_tasks)