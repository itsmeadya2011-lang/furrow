from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult, settings
from furrow.llm import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, workspace: Path | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.workspace = workspace or settings.workspace
        self.cycles = 0
        self.task_history: list[TaskModel] = []
        self.completed_tasks: list[TaskModel] = []
        self.failed_tasks: list[TaskModel] = []
        self.start_time = time.time()
        self.state_file = self.workspace / STATE_FILE

        if resume:
            self._load_state()

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("Starting Furrow with goal: %s", self.goal)
        try:
            while True:
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                logger.info("Starting cycle %d", self.cycles)
                await self._cycle()
                self._save_state()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info("Goal completed successfully")
                    break
                # Check max cycles
                if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                    console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Stopping.[/yellow]")
                    logger.warning("Max cycles reached")
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Saving state...[/yellow]")
            logger.info("Interrupted by user")
            self._save_state()
            raise
        except BaseException as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.exception("Uncaught exception in orchestrator")
            self._save_state()
            raise
        finally:
            # Clean up state file on successful completion
            if self._is_done() and self.state_file.exists():
                self.state_file.unlink()
                logger.info("Cleaned up state file")

    async def _cycle(self) -> None:
        # Show progress summary
        self._print_progress()

        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.goal, self._get_workspace_context())
            except BaseException as e:
                console.print(f"[red]Planning failed: {e}[/red]")
                logger.exception("Planning failed")
                return
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks
        pending_tasks = [t for t in plan.tasks if t.id not in {ct.id for ct in self.completed_tasks}]
        if not pending_tasks:
            console.print("[green]All planned tasks already completed.[/green]")
            return

        # Respect task dependencies - execute independent tasks in parallel
        execution_order = self._resolve_dependencies(pending_tasks)
        
        for batch in execution_order:
            with Status(f"[bold yellow]Executing {len(batch)} task(s)...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client, workspace=self.workspace).run()
                    for task in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    self.failed_tasks.append(task)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("Task %s failed: %s", task.id, result)
                else:
                    task.status = "completed"
                    task.result = result
                    self.completed_tasks.append(task)
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("Task %s completed", task.id)

        self.task_history.extend(pending_tasks)

        with Status("[bold yellow]Testing...", console=console):
            try:
                test_result = await TesterAgent(client=self.client).run(self.goal, pending_tasks)
            except BaseException as e:
                console.print(f"[red]Testing failed: {e}[/red]")
                logger.exception("Testing failed")
                test_result = TestResult(passed=False, summary=str(e), failures=[str(e)])

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed: %s", test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("Tests failed: %s", test_result.failures)
            # Update goal with failures for next cycle
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        # Check if all tasks from the last plan are completed
        if not self.task_history:
            return False
        pending_ids = {t.id for t in self.task_history if t.status != "completed"}
        if pending_ids:
            return False
        # If we have completed tasks and no failures in last cycle, we're done
        return bool(self.completed_tasks and not self.failed_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.task_history

    def _get_workspace_context(self) -> str:
        """Get context about the workspace for the planner."""
        try:
            files = self.client.list_files(self.workspace)
            return "Workspace files:\n" + "\n".join(f"  {f}" for f in files[:50])
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to list workspace files: %s", e)
            return "Workspace: (unable to list files)"

    def _print_progress(self) -> None:
        table = Table(title=f"Progress (Cycle {self.cycles})")
        table.add_column("Status", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Tasks", style="green")

        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        total = len(self.task_history)

        table.add_row("Completed", str(completed), ", ".join(t.id for t in self.completed_tasks[-5:]))
        table.add_row("Failed", str(failed), ", ".join(t.id for t in self.failed_tasks[-5:]))
        table.add_row("Total Planned", str(total), "")
        table.add_row("Elapsed", f"{time.time() - self.start_time:.1f}s", "")

        console.print(table)

    def _resolve_dependencies(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Resolve task dependencies and return execution batches."""
        executed: set[str] = set()
        batches = []

        while len(executed) < len(tasks):
            # Find tasks whose dependencies are all met
            ready = [
                t for t in tasks
                if t.id not in executed and all(dep in executed for dep in t.dependencies)
            ]
            if not ready:
                # Circular dependency or missing dependency - execute remaining
                remaining = [t for t in tasks if t.id not in executed]
                if remaining:
                    console.print("[yellow]Warning: Circular/missing dependencies, executing remaining tasks[/yellow]")
                    batches.append(remaining)
                break
            batches.append(ready)
            executed.update(t.id for t in ready)

        return batches

    def _save_state(self) -> None:
        """Save orchestrator state to file."""
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "start_time": self.start_time,
            "task_history": [t.model_dump() for t in self.task_history],
            "completed_tasks": [t.model_dump() for t in self.completed_tasks],
            "failed_tasks": [t.model_dump() for t in self.failed_tasks],
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug("State saved to %s", self.state_file)
        except OSError as e:
            console.print(f"[red]Failed to save state: {e}[/red]")
            logger.error("Failed to save state: %s", e)

    def _load_state(self) -> None:
        """Load orchestrator state from file."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            self.start_time = state.get("start_time", self.start_time)
            self.task_history = [TaskModel(**t) for t in state.get("task_history", [])]
            self.completed_tasks = [TaskModel(**t) for t in state.get("completed_tasks", [])]
            self.failed_tasks = [TaskModel(**t) for t in state.get("failed_tasks", [])]
            console.print(f"[green]Resumed from cycle {self.cycles}[/green]")
            logger.info("Resumed from cycle %d", self.cycles)
        except (OSError, json.JSONDecodeError) as e:
            console.print(f"[red]Failed to load state: {e}[/red]")
            logger.error("Failed to load state: %s", e)
