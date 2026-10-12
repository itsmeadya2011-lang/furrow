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
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, state_file: str | Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.state_file = Path(state_file) if state_file else Path(settings.workspace) / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.completed_task_ids = set(data.get("completed", []))
                self.cycles = data.get("cycles", 0)
                console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, {len(self.completed_task_ids)} completed[/dim]")
                logger.info("state_loaded", tasks=len(self.all_tasks), completed=len(self.completed_task_ids))
            except (OSError, json.JSONDecodeError) as e:
                console.print(f"[yellow]Failed to load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
                "completed": list(self.completed_task_ids),
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("state_saved", cycles=self.cycles, tasks=len(self.all_tasks))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            console.print(f"[yellow]Failed to save state: {e}[/yellow]")
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("furrow_started", goal=self.goal)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Stopping.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                break

    async def _cycle(self) -> None:
        # Get project context for planner
        project_files = self.client.list_files(settings.workspace)
        
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, project_files, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already completed tasks and respect dependencies
        runnable_tasks = self._get_runnable_tasks(plan.tasks)
        if not runnable_tasks:
            console.print("[yellow]All tasks completed or blocked on dependencies.[/yellow]")
            return

        console.print(f"[dim]Running {len(runnable_tasks)} of {len(plan.tasks)} tasks (respecting dependencies)[/dim]")
        logger.info("executing_tasks", runnable=len(runnable_tasks), total=len(plan.tasks))

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in runnable_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(runnable_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result if isinstance(result, str) else str(result)
                self.completed_task_ids.add(task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id, result=task.result[:100])

        # Update our task list with new tasks from plan
        for new_task in plan.tasks:
            if not any(t.id == new_task.id for t in self.all_tasks):
                self.all_tasks.append(new_task)
            else:
                # Update existing task
                for i, t in enumerate(self.all_tasks):
                    if t.id == new_task.id:
                        self.all_tasks[i] = new_task
                        break

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            # Add failure context to goal for next cycle
            self.goal = f"Original goal: {self.goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    def _get_runnable_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Return tasks that are not completed and have all dependencies satisfied."""
        runnable = []
        for task in tasks:
            if task.id in self.completed_task_ids:
                continue
            # Check if all dependencies are completed
            deps_satisfied = all(dep in self.completed_task_ids for dep in task.dependencies)
            if deps_satisfied:
                runnable.append(task)
        return runnable

    def _is_done(self) -> bool:
        # Check if all tasks in our tracked list are completed
        pending = [t for t in self.all_tasks if t.id not in self.completed_task_ids]
        failed = [t for t in self.all_tasks if t.status == "failed"]
        if failed:
            return False
        return not pending

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
