from __future__ import annotations

import asyncio
import json
import os
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.llm import LLMClient

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
console = Console()


@dataclass
class OrchestratorState:
    """Persisted state for the orchestrator across cycles."""
    goal: str
    all_tasks: list[TaskModel] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    cycles: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> str:
        return json.dumps({
            "goal": self.goal,
            "all_tasks": [t.model_dump() for t in self.all_tasks],
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "cycles": self.cycles,
            "started_at": self.started_at,
            "updated_at": datetime.now().isoformat(),
        }, indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "OrchestratorState":
        obj = json.loads(data)
        obj["all_tasks"] = [TaskModel(**t) for t in obj.get("all_tasks", [])]
        return cls(**obj)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: str | Path | None = None,
        max_cycles: int = 0,
    ) -> None:
        self.state_file = Path(state_file) if state_file else Path.cwd() / ".furrow_state.json"
        self.max_cycles = max_cycles
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.state = self._load_state(goal)
    
    def _load_state(self, goal: str) -> OrchestratorState:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = OrchestratorState.from_json(f.read())
                    console.print(f"[dim]Loaded state from {self.state_file}[/dim]")
                    return state
            except Exception as e:
                console.print(f"[yellow]Could not load state: {e}. Starting fresh.[/yellow]")
        return OrchestratorState(goal=goal)
    
    def _save_state(self) -> None:
        self.state.updated_at = datetime.now().isoformat()
        try:
            with open(self.state_file, "w") as f:
                f.write(self.state.to_json())
        except Exception as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
    
    async def run(self) -> None:
        logger.info("furrow_started", goal=self.state.goal, max_cycles=self.max_cycles)
        console.print(Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.state.goal}",
            title="Furrow"
        ))
        
        while True:
            if self.max_cycles and self.state.cycles >= self.max_cycles:
                logger.info("max_cycles_reached", cycles=self.state.cycles, max_cycles=self.max_cycles)
                console.print(f"[yellow]Reached max cycles ({self.max_cycles}). Stopping.[/yellow]")
                break
            
            self.state.cycles += 1
            logger.info("cycle_started", cycle=self.state.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.state.cycles} ═══[/bold cyan]")
            
            try:
                await self._cycle()
            except Exception as e:
                logger.error("cycle_failed", cycle=self.state.cycles, error=str(e), exc_info=True)
                console.print(f"[red]Cycle {self.state.cycles} failed: {e}[/red]")
                raise
            
            self._save_state()
            
            if self._is_done():
                logger.info("goal_completed", cycles=self.state.cycles, total_tasks=len(self.state.all_tasks))
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
    
    async def _cycle(self) -> None:
        # Plan
        logger.info("planning_started", cycle=self.state.cycles, goal=self.state.goal)
        with Status("[bold yellow]Planning...", console=console, spinner="dots"):
            plan = await self.planner.plan(self.state.goal)
        
        logger.info("planning_completed", cycle=self.state.cycles, task_count=len(plan.tasks))
        self._display_plan(plan)
        
        if not plan.tasks:
            logger.warning("no_tasks_planned", cycle=self.state.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return
        
        # Merge new tasks with existing state, avoiding duplicates
        new_task_ids = {t.id for t in plan.tasks}
        for task in plan.tasks:
            if task.id not in new_task_ids:
                continue
            existing = next((t for t in self.state.all_tasks if t.id == task.id), None)
            if not existing:
                self.state.all_tasks.append(task)
                logger.info("task_added", task_id=task.id, description=task.description)
        
        # Execute tasks respecting dependencies
        logger.info("execution_started", cycle=self.state.cycles, task_count=len(plan.tasks))
        await self._execute_tasks(plan.tasks)
        
        # Test
        logger.info("testing_started", cycle=self.state.cycles)
        with Status("[bold yellow]Testing...", console=console, spinner="dots") as status:
            test_result = await TesterAgent(client=self.client).run(self.state.goal, self.state.all_tasks)
        
        logger.info("testing_completed", cycle=self.state.cycles, passed=test_result.passed)
        self._display_test_result(test_result)
        
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.state.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.warning("tests_failed", cycle=self.state.cycles, failures=test_result.failures)
    
    def _display_plan(self, plan: Plan) -> None:
        table = Table(title="Plan", show_header=True, header_style="bold blue")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Files", style="dim")
        table.add_column("Dependencies", style="dim")
        
        for task in plan.tasks:
            table.add_row(
                task.id,
                task.description,
                ", ".join(task.files) if task.files else "any",
                ", ".join(task.dependencies) if task.dependencies else "none"
            )
        
        console.print(table)
        console.print(f"[dim]Rationale: {plan.rationale}[/dim]")
    
    def _display_test_result(self, result: TestResult) -> None:
        style = "green" if result.passed else "red"
        console.print(f"[{style}]Tests: {'PASSED' if result.passed else 'FAILED'}[/{style}]")
        console.print(f"  Summary: {result.summary}")
        if result.failures:
            for failure in result.failures:
                console.print(f"  [red]• {failure}[/red]")
    
    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks in dependency order, running independent tasks in parallel."""
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        completed = set()
        failed = set()
        
        # Filter to only tasks that are pending
        pending = [t for t in tasks if t.status == "pending" and t.id not in completed]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            while pending:
                # Find tasks whose dependencies are all completed
                ready = [
                    t for t in pending
                    if all(dep in completed for dep in t.dependencies)
                ]
                
                if not ready:
                    # Check for circular dependencies or unmet deps
                    unmet = [t for t in pending if any(dep not in completed for dep in t.dependencies)]
                    if unmet:
                        logger.warning("waiting_for_dependencies", unmet_tasks=[t.id for t in unmet])
                        console.print(f"[yellow]Waiting for dependencies: {[t.id for t in unmet]}[/yellow]")
                    break
                
                logger.info("executing_parallel_tasks", task_ids=[t.id for t in ready])
                
                # Run ready tasks in parallel
                task_refs = []
                for task in ready:
                    task_ref = progress.add_task(f"Task {task.id}: {task.description[:50]}", total=None)
                    task_refs.append((task, task_ref))
                
                workers = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task, _ in task_refs
                ]
                
                results = await asyncio.gather(*workers, return_exceptions=True)
                
                for (task, task_ref), result in zip(task_refs, results):
                    progress.update(task_ref, completed=True)
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        failed.add(task.id)
                        self.state.failed_tasks.append(task.id)
                        logger.error("task_failed", task_id=task.id, error=str(result))
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        completed.add(task.id)
                        self.state.completed_tasks.append(task.id)
                        logger.info("task_completed", task_id=task.id, result=result[:100] if result else "")
                        console.print(f"[green]Task {task.id} completed[/green]")
                
                # Update pending
                pending = [t for t in pending if t.id not in completed and t.id not in failed]
    
    def _is_done(self) -> bool:
        if not self.state.all_tasks:
            return False
        
        all_task_ids = {t.id for t in self.state.all_tasks}
        completed_ids = set(self.state.completed_tasks)
        failed_ids = set(self.state.failed_tasks)
        
        # If any task failed and not fixed, not done
        if failed_ids:
            return False
        
        # If all tasks completed, done
        if completed_ids >= all_task_ids:
            return True
        
        return False
    
    def _get_tasks(self) -> list[TaskModel]:
        return self.state.all_tasks
    
    def get_progress_summary(self) -> dict[str, Any]:
        """Get a summary of current progress."""
        return {
            "goal": self.state.goal,
            "cycles": self.state.cycles,
            "total_tasks": len(self.state.all_tasks),
            "completed": len(self.state.completed_tasks),
            "failed": len(self.state.failed_tasks),
            "pending": len(self.state.all_tasks) - len(self.state.completed_tasks) - len(self.state.failed_tasks),
            "started_at": self.state.started_at,
            "updated_at": self.state.updated_at,
        }