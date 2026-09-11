from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        state_path: Path | None = None,
        on_output: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.state_path = state_path or (self.config.workspace / ".furrow_state.json")
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.on_output = on_output
        self._feedback_queue: asyncio.Queue[str] = asyncio.Queue()
        self._paused = False
        self._load_state()

    def _emit(self, message: str) -> None:
        """Emit a message to the output callback or console."""
        if self.on_output:
            self.on_output(message)
        else:
            console.print(message)

    def _log(self, level: str, event: str, **kwargs: Any) -> None:
        """Log a structured message."""
        getattr(logger, level.lower())(event, **kwargs)

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.completed_task_ids = set(data.get("completed_task_ids", []))
                for task_data in data.get("all_tasks", []):
                    task = TaskModel(**task_data)
                    self.all_tasks.append(task)
                self._emit(f"[green]Loaded state from {self.state_path}[/green]")
                self._log("info", "state_loaded", path=str(self.state_path), cycles=self.cycles)
            except Exception as e:
                self._emit(f"[yellow]Could not load state: {e}[/yellow]")
                self._log("warning", "state_load_failed", error=str(e))

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "completed_task_ids": list(self.completed_task_ids),
            "all_tasks": [t.model_dump() for t in self.all_tasks],
        }
        self.state_path.write_text(json.dumps(data, indent=2))
        self._log("debug", "state_saved", path=str(self.state_path))

    async def run(self) -> None:
        self._log("info", "orchestrator_started", goal=self.goal)
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.config.max_cycles and self.cycles >= self.config.max_cycles:
                self._log("info", "max_cycles_reached", max_cycles=self.config.max_cycles)
                self._emit(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            self._log("info", "cycle_started", cycle=self.cycles)
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._log("info", "goal_complete", cycles=self.cycles)
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Check for user feedback before planning
        await self._process_feedback()

        self._log("info", "planning_started", cycle=self.cycles)
        self._emit("[bold yellow]Planning...[/bold yellow]")
        plan = await self.planner.plan(self.goal, self.all_tasks, self.completed_task_ids)
        self._log("info", "plan_created", cycle=self.cycles, task_count=len(plan.tasks))
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._log("info", "no_tasks_planned", cycle=self.cycles)
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        tasks_to_run = self._get_executable_tasks(plan.tasks)
        if not tasks_to_run:
            self._log("info", "no_executable_tasks", cycle=self.cycles)
            self._emit("[yellow]No executable tasks (dependencies not met). Waiting...[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
            async with semaphore:
                result = await WorkerAgent(task=task, client=self.client, settings=self.config).run()
                return task, result

        self._log("info", "execution_started", cycle=self.cycles, task_count=len(tasks_to_run))
        self._emit("[bold yellow]Executing tasks in parallel...[/bold yellow]")
        worker_tasks = [run_with_semaphore(task) for task in tasks_to_run]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._log("error", "task_failed", task_id=task.id, error=str(result))
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                self._log("info", "task_completed", task_id=task.id)
                self._emit(f"[green]Task {task.id} completed[/green]")

        existing_ids = {t.id for t in self.all_tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.all_tasks.append(task)

        self._log("info", "testing_started", cycle=self.cycles)
        self._emit("[bold yellow]Testing...[/bold yellow]")
        test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, self.all_tasks)

        if test_result.passed:
            self._log("info", "tests_passed", cycle=self.cycles, summary=test_result.summary)
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._log("warning", "tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _process_feedback(self) -> None:
        """Process any queued user feedback before planning."""
        while not self._feedback_queue.empty():
            feedback = await self._feedback_queue.get()
            self._log("info", "user_feedback_received", cycle=self.cycles, feedback=feedback)
            self._emit(f"[bold magenta]User feedback: {feedback}[/bold magenta]")
            # Incorporate feedback into goal for next planning cycle
            self.goal = f"{self.goal}\n\nUser feedback: {feedback}"

    def add_feedback(self, feedback: str) -> None:
        """Add user feedback to be processed in the next cycle."""
        self._feedback_queue.put_nowait(feedback)

    def pause(self) -> None:
        """Pause the orchestrator after the current cycle."""
        self._paused = True
        self._emit("[yellow]Pausing after current cycle...[/yellow]")

    def resume(self) -> None:
        """Resume the orchestrator."""
        self._paused = False
        self._emit("[green]Resuming...[/green]")

    def _get_executable_tasks(self, new_tasks: list[TaskModel]) -> list[TaskModel]:
        executable = []
        for task in new_tasks:
            if task.id in self.completed_task_ids:
                continue
            deps_met = all(dep in self.completed_task_ids for dep in task.dependencies)
            if deps_met:
                executable.append(task)
        return executable

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        pending = [t for t in self.all_tasks if t.id not in self.completed_task_ids]
        if not pending:
            return True
        failed = [t for t in self.all_tasks if t.status == "failed"]
        if failed:
            return False
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks