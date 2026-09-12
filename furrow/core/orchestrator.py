from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | str | None = None,
        resume: bool = False,
    ) -> None:
        self.state_file = Path(state_file) if state_file else settings.workspace / ".furrow_state.json"

        # Load existing state if resuming and file exists
        loaded_state: dict[str, Any] | None = None
        if resume and self.state_file.exists():
            loaded_state = self.load_state()

        if loaded_state:
            self.goal = loaded_state.get("goal", goal)
            self.cycles = loaded_state.get("cycles", 0)
            self.all_tasks = self._deserialize_tasks(loaded_state.get("all_tasks", []))
            self.task_status_changes = loaded_state.get("task_status_changes", {})
            self.last_test_failures = loaded_state.get("last_test_failures", [])
            self.completed_per_cycle = loaded_state.get("completed_per_cycle", [])
        else:
            self.goal = goal
            self.cycles = 0
            self.all_tasks = []
            self.task_status_changes = {}
            self.last_test_failures = []
            self.completed_per_cycle = []

        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)

    def load_state(self) -> dict[str, Any] | None:
        """Load state from the state file. Returns the parsed dict or None on failure."""
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            console.print(f"[yellow]Could not load state from {self.state_file}: {e}[/yellow]")
            return None

    def save_state(self) -> None:
        """Persist current state to the state file."""
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "all_tasks": self._serialize_tasks(self.all_tasks),
            "last_test_failures": self.last_test_failures,
            "completed_per_cycle": self.completed_per_cycle,
            "task_status_changes": self.task_status_changes,
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except OSError as e:
            console.print(f"[yellow]Could not save state to {self.state_file}: {e}[/yellow]")

    def _serialize_tasks(self, tasks: list[TaskModel]) -> list[dict[str, Any]]:
        """Convert a list of TaskModel objects to JSON-serializable dicts."""
        return [task.model_dump() for task in tasks]

    def _deserialize_tasks(self, data: list[dict[str, Any]]) -> list[TaskModel]:
        """Convert a list of dicts back to TaskModel objects."""
        return [TaskModel(**task_data) for task_data in data]

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self.save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        completed_tasks = [t.description for t in self.all_tasks if t.status == "completed"]
        failed_tasks = [t.description for t in self.all_tasks if t.status == "failed"]

        planning_goal = self.goal
        if self.last_test_failures:
            planning_goal = (
                f"{self.goal}\n\nPrevious test failures to address:\n"
                + "\n".join(f"• {f}" for f in self.last_test_failures)
            )

        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(
                planning_goal,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                last_test_failures=self.last_test_failures,
                cycle_number=self.cycles,
            )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks and their status changes
        for task in plan.tasks:
            self.all_tasks.append(task)
            self.task_status_changes.setdefault(task.id, []).append(("pending", task.status))

        # Execute tasks in parallel, limited by max_parallel_tasks
        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)

        async def _run_with_semaphore(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_completed: list[str] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.task_status_changes.setdefault(task.id, []).append(("pending", "failed"))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                cycle_completed.append(task.id)
                self.task_status_changes.setdefault(task.id, []).append(("pending", "completed"))
                console.print(f"[green]Task {task.id} completed[/green]")

        self.completed_per_cycle.append(cycle_completed)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.last_test_failures = []
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            # Store failures as context for the next planning cycle instead of overwriting the goal
            self.last_test_failures = list(test_result.failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _is_done(self) -> bool:
        # Enforce max_cycles when set (> 0)
        if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
            return True

        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        # Any failed tasks means we are not done
        if failed > 0:
            return False

        # All tasks completed and no failures → done
        if tasks and completed >= len(tasks):
            return True

        return False