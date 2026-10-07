from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

# Callback type for streaming updates
ProgressCallback = Callable[[dict], None]


@dataclass
class CycleState:
    """State for a single cycle."""
    cycle_num: int
    goal: str
    plan: Optional[Plan] = None
    tasks: list[TaskModel] = field(default_factory=list)
    test_result: Optional[TestResult] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None


@dataclass
class OrchestratorState:
    """Persisted state for the orchestrator."""
    goal: str
    cycles: list[CycleState] = field(default_factory=list)
    all_tasks: dict[str, TaskModel] = field(default_factory=dict)
    completed_task_ids: set[str] = field(default_factory=set)
    failed_task_ids: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "goal": self.goal,
            "cycles": [
                {
                    "cycle_num": c.cycle_num,
                    "goal": c.goal,
                    "plan": c.plan.model_dump() if c.plan else None,
                    "tasks": [t.model_dump() for t in c.tasks],
                    "test_result": c.test_result.model_dump() if c.test_result else None,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "error": c.error,
                }
                for c in self.cycles
            ],
            "all_tasks": {k: v.model_dump() for k, v in self.all_tasks.items()},
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        })

    @classmethod
    def from_json(cls, data: str) -> OrchestratorState:
        obj = json.loads(data)
        state = cls(goal=obj["goal"])
        state.completed_task_ids = set(obj.get("completed_task_ids", []))
        state.failed_task_ids = set(obj.get("failed_task_ids", []))
        state.created_at = obj.get("created_at", time.time())
        state.updated_at = obj.get("updated_at", time.time())

        for c in obj.get("cycles", []):
            cycle = CycleState(
                cycle_num=c["cycle_num"],
                goal=c["goal"],
                plan=Plan(**c["plan"]) if c.get("plan") else None,
                tasks=[TaskModel(**t) for t in c.get("tasks", [])],
                test_result=TestResult(**c["test_result"]) if c.get("test_result") else None,
                start_time=c.get("start_time", time.time()),
                end_time=c.get("end_time"),
                error=c.get("error"),
            )
            state.cycles.append(cycle)

        for k, v in obj.get("all_tasks", {}).items():
            state.all_tasks[k] = TaskModel(**v)

        return state


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state: OrchestratorState | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.state = state or OrchestratorState(goal=goal)
        self.goal = self.state.goal
        self.cycles = len(self.state.cycles)
        self._max_cycles = self.settings.max_cycles or 0
        self._state_file = self.settings.workspace / ".furrow_state.json"
        self._on_progress = on_progress

    def _emit(self, event: dict) -> None:
        """Emit a progress event to callback if registered."""
        if self._on_progress:
            try:
                self._on_progress(event)
            except Exception:
                pass  # Silently ignore callback errors

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if self.state.cycles:
            console.print(f"[cyan]Resuming from cycle {self.cycles + 1}[/cyan]")

        self._emit({"type": "start", "goal": self.goal, "cycle": self.cycles + 1})

        try:
            while True:
                if self._max_cycles and self.cycles >= self._max_cycles:
                    console.print(f"[yellow]Max cycles ({self._max_cycles}) reached. Stopping.[/yellow]")
                    break

                self.cycles += 1
                await self._cycle()

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    self._emit({"type": "complete", "goal": self.goal, "cycles": self.cycles})
                    break

                self._save_state()

        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            self._save_state()
            self._emit({"type": "interrupted", "goal": self.goal, "cycles": self.cycles})
            raise

    async def _cycle(self) -> None:
        cycle_state = CycleState(cycle_num=self.cycles, goal=self.goal)
        self.state.cycles.append(cycle_state)
        console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
        self._emit({"type": "cycle_start", "cycle": self.cycles, "goal": self.goal})

        try:
            # Planning
            self._emit({"type": "planning", "cycle": self.cycles})
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self.state)
            cycle_state.plan = plan
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
            self._emit({"type": "plan", "cycle": self.cycles, "plan": plan.model_dump()})

            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                cycle_state.end_time = time.time()
                self._emit({"type": "cycle_end", "cycle": self.cycles, "tasks": 0})
                return

            # Register new tasks
            for task in plan.tasks:
                if task.id not in self.state.all_tasks:
                    self.state.all_tasks[task.id] = task
                cycle_state.tasks.append(self.state.all_tasks[task.id])

            self._emit({"type": "tasks", "cycle": self.cycles, "tasks": [t.model_dump() for t in cycle_state.tasks]})

            # Execution
            self._emit({"type": "executing", "cycle": self.cycles})
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    for task in cycle_state.tasks
                    if task.id not in self.state.completed_task_ids and task.id not in self.state.failed_task_ids
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(cycle_state.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self.state.failed_task_ids.add(task.id)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    self._emit({"type": "task_failed", "cycle": self.cycles, "task_id": task.id, "error": str(result)})
                else:
                    task.status = "completed"
                    task.result = result
                    self.state.completed_task_ids.add(task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
                    self._emit({"type": "task_completed", "cycle": self.cycles, "task_id": task.id, "result": result})

            # Testing
            self._emit({"type": "testing", "cycle": self.cycles})
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                    self.goal, cycle_state.tasks
                )
            cycle_state.test_result = test_result

            if test_result.passed:
                console.print(f"[green]Tests passed: {test_result.summary}[/green]")
                self._emit({"type": "tests_passed", "cycle": self.cycles, "summary": test_result.summary})
            else:
                console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    console.print(f"  • {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
                self._emit({"type": "tests_failed", "cycle": self.cycles, "summary": test_result.summary, "failures": test_result.failures})
                # Append fix context to goal instead of replacing
                self.goal = self._update_goal_with_failures(self.goal, test_result.failures)

        except Exception as e:
            cycle_state.error = str(e)
            console.print(f"[red]Cycle {self.cycles} error: {e}[/red]")
            self._emit({"type": "cycle_error", "cycle": self.cycles, "error": str(e)})
            raise
        finally:
            cycle_state.end_time = time.time()
            self.state.updated_at = time.time()
            self._emit({"type": "cycle_end", "cycle": self.cycles})

    def _update_goal_with_failures(self, goal: str, failures: list[str]) -> str:
        """Append test failures to goal for next cycle."""
        if "Fix failing tests:" in goal:
            # Already has failures, just append new ones
            return goal + "\n" + "\n".join(failures)
        return f"Fix failing tests:\n" + "\n".join(failures) + "\n\nOriginal goal:\n" + goal

    def _is_done(self) -> bool:
        """Check if all tasks are completed and tests pass."""
        if not self.state.all_tasks:
            return False

        # Check if there are any pending or failed tasks
        for task in self.state.all_tasks.values():
            if task.status != "completed":
                return False

        # Check if last test passed
        if self.state.cycles:
            last_test = self.state.cycles[-1].test_result
            if last_test and not last_test.passed:
                return False

        return True

    def _save_state(self) -> None:
        """Save state to disk for resume capability."""
        try:
            self._state_file.write_text(self.state.to_json())
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save state: {e}[/yellow]")

    def get_status_table(self) -> Table:
        """Get a rich table showing current status."""
        table = Table(title="Furrow Status")
        table.add_column("Cycle", style="cyan")
        table.add_column("Tasks", style="white")
        table.add_column("Completed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Tests", style="yellow")

        for cycle in self.state.cycles:
            completed = sum(1 for t in cycle.tasks if t.status == "completed")
            failed = sum(1 for t in cycle.tasks if t.status == "failed")
            test_status = "pass" if cycle.test_result and cycle.test_result.passed else "fail"
            table.add_row(
                str(cycle.cycle_num),
                str(len(cycle.tasks)),
                str(completed),
                str(failed),
                test_status,
            )

        return table