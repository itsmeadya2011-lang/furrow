from __future__ import annotations

import asyncio
import json
import time

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, State, TaskModel
from furrow.llm import LLMClient
from furrow.logging import (
    get_logger,
    log_cycle_complete,
    log_cycle_start,
    log_task_complete,
    log_task_failed,
    log_task_start,
)

console = Console()
logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        resume: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.state_file = self.settings.state_file
        self.max_cycles = self.settings.max_cycles
        self.max_parallel_tasks = self.settings.max_parallel_tasks

        if resume and self.state_file.exists():
            self.state = self._load_state()
            # Update goal if provided
            if goal and goal != self.state.goal:
                self.state.goal = goal
                self._add_history(f"Goal updated to: {goal}")
        else:
            self.state = State(goal=goal)
            self._add_history(f"Started with goal: {goal}")

    def _load_state(self) -> State:
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            return State(**data)
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[yellow]Could not load state: {e}. Starting fresh.[/yellow]")
            return State(goal=self.state.goal if hasattr(self, "state") else "")

    def _save_state(self) -> None:
        self.state.updated_at = time.time()
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state.model_dump(), f, indent=2)
        except OSError as e:
            console.print(f"[red]Failed to save state: {e}[/red]")

    def _add_history(self, entry: str) -> None:
        self.state.history.append(f"[{time.strftime('%H:%M:%S')}] {entry}")
        if len(self.state.history) > 100:
            self.state.history = self.state.history[-100:]

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.state.goal}", title="Furrow")
        )
        logger.info("orchestrator_started", goal=self.state.goal)

        while not self._is_done():
            self.state.cycles += 1
            cycle_num = self.state.cycles
            console.print(f"\n[bold cyan]═══ Cycle {cycle_num} ═══[/bold cyan]")
            log_cycle_start(logger, cycle_num, self.state.goal)

            if self.max_cycles > 0 and cycle_num > self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Stopping.[/yellow]")
                logger.warning("max_cycles_reached", max_cycles=self.max_cycles)
                break

            await self._cycle()
            self._save_state()

            completed = len(self.state.completed_tasks)
            log_cycle_complete(logger, cycle_num, completed, True)

        console.print("[bold green]Goal complete. Halting.[/bold green]")
        logger.info("orchestrator_completed", cycles=self.state.cycles)
        self._print_summary()

    async def _cycle(self) -> None:
        cycle_num = self.state.cycles
        self._add_history(f"Cycle {cycle_num} started")

        # Plan
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.state.goal, self._get_context())

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._add_history(f"Planned {len(plan.tasks)} tasks: {plan.rationale}")
        logger.info(
            "plan_created", cycle=cycle_num, task_count=len(plan.tasks), rationale=plan.rationale
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._add_history("No tasks planned")
            return

        # Filter tasks: only run pending tasks without unmet dependencies
        runnable_tasks = self._get_runnable_tasks(plan.tasks)
        if not runnable_tasks:
            console.print("[yellow]No runnable tasks (all blocked by dependencies).[/yellow]")
            self._add_history("No runnable tasks")
            return

        # Update state with new tasks (merge with existing)
        self._merge_tasks(plan.tasks)

        # Execute runnable tasks in parallel (respect max_parallel_tasks)
        total_completed = 0
        for i in range(0, len(runnable_tasks), self.max_parallel_tasks):
            batch = runnable_tasks[i : i + self.max_parallel_tasks]
            completed = await self._execute_batch(batch)
            total_completed += completed

        # Test
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.state.goal, self.state.tasks
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._add_history(f"Tests passed: {test_result.summary}")
            logger.info("tests_passed", cycle=cycle_num, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._add_history(
                f"Tests failed: {test_result.summary}; failures: {test_result.failures}"
            )
            logger.warning(
                "tests_failed",
                cycle=cycle_num,
                summary=test_result.summary,
                failures=test_result.failures,
            )
            # Update goal to focus on fixing failures
            self.state.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_context(self) -> str:
        """Build context string for planner including project files and history."""
        parts = []
        if self.state.tasks:
            parts.append("Current tasks:")
            for t in self.state.tasks:
                parts.append(f"  - {t.id}: {t.description} [{t.status}]")
        if self.state.history:
            parts.append("\nRecent history:")
            for h in self.state.history[-10:]:
                parts.append(f"  {h}")
        # Add project structure
        files = self.client.list_files(self.settings.workspace)
        if files:
            parts.append(f"\nProject files ({len(files)}):")
            for f in sorted(files)[:50]:
                parts.append(f"  {f}")
            if len(files) > 50:
                parts.append(f"  ... and {len(files) - 50} more")
        return "\n".join(parts)

    def _get_runnable_tasks(self, new_tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks that are pending and have all dependencies met."""
        # Build a map of all tasks (existing + new)
        all_tasks = {t.id: t for t in self.state.tasks}
        for t in new_tasks:
            all_tasks[t.id] = t

        runnable = []
        for task in new_tasks:
            if task.status != "pending":
                continue
            # Check dependencies
            deps_met = all(
                all_tasks.get(dep_id, TaskModel(id=dep_id, description="")).status == "completed"
                for dep_id in task.dependencies
            )
            if deps_met:
                runnable.append(task)
        return runnable

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        """Merge new tasks into state, preserving status of existing tasks."""
        existing = {t.id: t for t in self.state.tasks}
        for task in new_tasks:
            if task.id in existing and existing[task.id].status in ("completed", "failed"):
                task.status = existing[task.id].status
                task.result = existing[task.id].result
            existing[task.id] = task
        self.state.tasks = list(existing.values())

    async def _execute_batch(self, tasks: list[TaskModel]) -> int:
        completed_count = 0
        with Status(f"[bold yellow]Executing {len(tasks)} tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(tasks, results):
            log_task_start(logger, task.id, task.description)
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self.state.failed_tasks.append(task.id)
                self._add_history(f"Task {task.id} failed: {result}")
                log_task_failed(logger, task.id, str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self.state.completed_tasks.append(task.id)
                self._add_history(f"Task {task.id} completed: {result[:100]}")
                log_task_complete(logger, task.id, result)
                completed_count += 1

        return completed_count

    def _is_done(self) -> bool:
        if not self.state.tasks:
            return False  # No tasks yet, need at least one cycle

        completed = sum(1 for t in self.state.tasks if t.status == "completed")
        failed = sum(1 for t in self.state.tasks if t.status == "failed")
        pending = sum(1 for t in self.state.tasks if t.status == "pending")

        # If there are failed tasks that haven't been fixed, not done
        if failed > 0 and self.state.goal.startswith("Fix failing tests"):
            return False

        # Done if all tasks completed and no pending tasks
        return pending == 0 and completed > 0

    def _print_summary(self) -> None:
        table = Table(title="Furrow Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Cycles", str(self.state.cycles))
        table.add_row("Total Tasks", str(len(self.state.tasks)))
        table.add_row("Completed", str(len(self.state.completed_tasks)))
        table.add_row("Failed", str(len(self.state.failed_tasks)))
        table.add_row("Goal", self.state.goal[:80] + ("..." if len(self.state.goal) > 80 else ""))

        console.print(table)
