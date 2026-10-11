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
from furrow.config import Settings, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)
console = Console()


class OrchestratorState:
    """Persistable state for the orchestrator."""
    
    def __init__(self) -> None:
        self.goal: str = ""
        self.cycles: int = 0
        self.all_tasks: list[dict[str, Any]] = []
        self.completed_task_ids: set[str] = set()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "all_tasks": self.all_tasks,
            "completed_task_ids": list(self.completed_task_ids),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestratorState:
        state = cls()
        state.goal = data.get("goal", "")
        state.cycles = data.get("cycles", 0)
        state.all_tasks = data.get("all_tasks", [])
        state.completed_task_ids = set(data.get("completed_task_ids", []))
        return state


class Orchestrator:
    def __init__(
        self, 
        goal: str, 
        client: LLMClient | None = None, 
        orchestrator_settings: Settings | None = None,
        state_path: Path | None = None
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.settings = orchestrator_settings or settings
        self._all_tasks: list[Any] = []
        self._completed_task_ids: set[str] = set()
        self._state_path = state_path or (Path.cwd() / ".furrow_state.json")
        
        # Load previous state if exists
        self._load_state()

    def _load_state(self) -> None:
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    data = json.load(f)
                state = OrchestratorState.from_dict(data)
                self.goal = state.goal or self.goal
                self.cycles = state.cycles
                self._completed_task_ids = state.completed_task_ids
                # Restore tasks from dict
                from furrow.config import TaskModel
                self._all_tasks = [TaskModel(**t) for t in state.all_tasks]
                console.print(f"[dim]Loaded state from {self._state_path} (cycle {self.cycles})[/dim]")
                logger.info("state_loaded", path=str(self._state_path), cycle=self.cycles)
            except (OSError, json.JSONDecodeError) as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")
                logger.warning("state_load_failed", path=str(self._state_path), error=str(e))

    def _save_state(self) -> None:
        state = OrchestratorState()
        state.goal = self.goal
        state.cycles = self.cycles
        state.all_tasks = [t.model_dump() for t in self._all_tasks]
        state.completed_task_ids = self._completed_task_ids
        try:
            with open(self._state_path, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
            logger.debug("state_saved", path=str(self._state_path), cycle=self.cycles)
        except OSError as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
            logger.warning("state_save_failed", path=str(self._state_path), error=str(e))

    async def run(self) -> None:
        logger.info("orchestrator_started", goal=self.goal, max_cycles=self.settings.max_cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            
            # Check max cycles limit
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break
                
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", cycles=self.cycles)
                # Clean up state file on completion
                if self._state_path.exists():
                    self._state_path.unlink()
                break

    async def _cycle(self) -> None:
        logger.debug("planning_started", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        # Add new tasks to our tracked list
        for task in plan.tasks:
            if task.id not in self._completed_task_ids:
                self._all_tasks.append(task)

        # Limit parallel tasks
        tasks_to_run = [t for t in plan.tasks if t.status == "pending"]
        if len(tasks_to_run) > self.settings.max_parallel_tasks:
            tasks_to_run = tasks_to_run[:self.settings.max_parallel_tasks]
            console.print(f"[yellow]Limiting to {self.settings.max_parallel_tasks} parallel tasks[/yellow]")
            logger.warning("parallel_tasks_limited", requested=len(plan.tasks), limit=self.settings.max_parallel_tasks)

        logger.info("execution_started", task_count=len(tasks_to_run))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(tasks_to_run, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = str(result) if result is not None else None
                self._completed_task_ids.add(task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        logger.debug("testing_started")
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._all_tasks:
            return False
        
        completed = sum(1 for t in self._all_tasks if t.status == "completed")
        failed = sum(1 for t in self._all_tasks if t.status == "failed")
        pending = sum(1 for t in self._all_tasks if t.status == "pending")
        
        if failed > 0 or pending > 0:
            return False
        return completed >= len(self._all_tasks)

    def _get_tasks(self) -> list[Any]:
        return self._all_tasks
