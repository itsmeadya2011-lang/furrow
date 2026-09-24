from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, Settings, settings
from furrow.llm import LLMClient

# Structured logger
logger = structlog.get_logger()

# Rich console for CLI output
console = Console()

STATE_FILE = ".furrow_state.json"

# Event types for callbacks
EventCallback = Callable[[str, dict], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        orchestrator_settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=orchestrator_settings)
        self.settings = orchestrator_settings or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[dict[str, Any]] = []  # Track all tasks across cycles
        self.completed_task_ids: set[str] = set()
        self.failed_task_ids: set[str] = set()
        self.on_event = on_event or (lambda event, data: None)
        self._load_state()

    def _emit(self, event: str, data: dict) -> None:
        """Emit an event to the callback and log it."""
        logger.info(event, **data)
        self.on_event(event, data)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_path = self.settings.workspace / STATE_FILE
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.all_tasks = state.get("all_tasks", [])
                self.completed_task_ids = set(state.get("completed_task_ids", []))
                self.failed_task_ids = set(state.get("failed_task_ids", []))
                self.cycles = state.get("cycles", 0)
                self._emit("state_loaded", {"tasks": len(self.all_tasks), "cycle": self.cycles})
            except Exception as e:
                logger.warning("could_not_load_state", error=str(e))

    def _save_state(self) -> None:
        """Persist state to disk."""
        state_path = self.settings.workspace / STATE_FILE
        try:
            state = {
                "goal": self.goal,
                "all_tasks": self.all_tasks,
                "completed_task_ids": list(self.completed_task_ids),
                "failed_task_ids": list(self.failed_task_ids),
                "cycles": self.cycles,
            }
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            self._emit("state_saved", {"path": str(state_path)})
        except Exception as e:
            logger.warning("could_not_save_state", error=str(e))

    def _get_ready_tasks(self, tasks: list[Any]) -> list[Any]:
        """Filter tasks whose dependencies are all completed."""
        ready = []
        for task in tasks:
            deps = task.dependencies or []
            if all(dep in self.completed_task_ids for dep in deps):
                ready.append(task)
            else:
                missing = [d for d in deps if d not in self.completed_task_ids]
                self._emit("task_waiting", {"task_id": task.id, "waiting_for": missing})
        return ready

    async def run(self) -> None:
        self._emit("started", {"goal": self.goal})
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            
            # Check max cycles
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                msg = f"Max cycles ({self.settings.max_cycles}) reached. Stopping."
                self._emit("max_cycles_reached", {"max_cycles": self.settings.max_cycles})
                console.print(f"[yellow]{msg}[/yellow]")
                break
            
            self._emit("cycle_start", {"cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            
            if self._is_done():
                self._emit("completed", {"cycles": self.cycles})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            
            self._save_state()
        
        self._emit("stopped", {"cycles": self.cycles})

    async def _cycle(self) -> None:
        # Get project context for planner
        project_files = self.client.list_files(self.settings.workspace)
        
        self._emit("planning_start", {"cycle": self.cycles})
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, project_files, self.all_tasks)
        self._emit("planning_complete", {"cycle": self.cycles, "tasks": len(plan.tasks)})
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("no_tasks_planned", {"cycle": self.cycles})
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed/failed tasks
        new_tasks = [t for t in plan.tasks if t.id not in self.completed_task_ids and t.id not in self.failed_task_ids]
        
        if not new_tasks:
            self._emit("all_tasks_done", {"cycle": self.cycles})
            console.print("[yellow]All planned tasks already completed or failed.[/yellow]")
            return

        # Track new tasks
        for task in new_tasks:
            task_dict = task.model_dump()
            existing = next((t for t in self.all_tasks if t["id"] == task.id), None)
            if existing:
                existing.update(task_dict)
            else:
                self.all_tasks.append(task_dict)

        # Execute tasks respecting dependencies
        await self._execute_with_dependencies(new_tasks)

        self._emit("testing_start", {"cycle": self.cycles})
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            self._emit("tests_passed", {"summary": test_result.summary})
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit("tests_failed", {"summary": test_result.summary, "failures": test_result.failures})
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Append failures to goal for next cycle
            self.goal = f"{self.goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_with_dependencies(self, tasks: list[Any]) -> None:
        """Execute tasks in parallel batches respecting dependencies."""
        remaining = {t.id: t for t in tasks}
        max_parallel = self.settings.max_parallel_tasks
        
        while remaining:
            # Get tasks ready to run (dependencies satisfied)
            ready = self._get_ready_tasks(list(remaining.values()))
            
            if not ready:
                # Check for circular dependencies or blocked tasks
                blocked = list(remaining.values())
                self._emit("deadlock_detected", {"blocked_tasks": [t.id for t in blocked]})
                console.print("[red]Deadlock detected: tasks waiting on unmet dependencies[/red]")
                for task in blocked:
                    task.status = "failed"
                    task.result = "Deadlock: unmet dependencies"
                    self.failed_task_ids.add(task.id)
                break

            # Limit parallel execution
            batch = ready[:max_parallel]
            if len(ready) > max_parallel:
                self._emit("tasks_limited", {"ready": len(ready), "max_parallel": max_parallel})
                console.print(f"[yellow]Running {max_parallel} of {len(ready)} ready tasks[/yellow]")

            self._emit("execution_start", {"cycle": self.cycles, "tasks": len(batch)})
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                worker_tasks = [
                    WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    for task in batch
                ]
                results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                del remaining[task.id]
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self.failed_task_ids.add(task.id)
                    self._emit("task_failed", {"task_id": task.id, "error": str(result)})
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self.completed_task_ids.add(task.id)
                    self._emit("task_completed", {"task_id": task.id, "result": result})
                    console.print(f"[green]Task {task.id} completed[/green]")

    def _is_done(self) -> bool:
        # Check if there are any pending tasks (not completed, not failed)
        pending = [t for t in self.all_tasks if t["id"] not in self.completed_task_ids and t["id"] not in self.failed_task_ids]
        
        # If there are failed tasks, not done
        if self.failed_task_ids:
            return False
        
        # If no pending tasks and we have completed tasks, we're done
        if not pending and self.completed_task_ids:
            return True
        
        # If no tasks at all yet, not done (need at least one cycle)
        if not self.all_tasks:
            return False
            
        return False

    def _get_tasks(self) -> list[Any]:
        # Return task objects for compatibility
        return self.all_tasks