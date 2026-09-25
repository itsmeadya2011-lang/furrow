from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.original_goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.task_results: dict[str, str] = {}
        self._completed_task_ids: set[str] = set()

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal, max_cycles=self.settings.max_cycles)
        
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles)
                break
                
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            
            await self._cycle()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        # Plan phase
        with Status("[bold yellow]Planning...", console=console) as _:
            plan = await self.planner.plan(self.goal, self.tasks, self.task_results)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_received", cycle=self.cycles, num_tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        # Filter out already completed tasks
        new_tasks = [t for t in plan.tasks if t.id not in self._completed_task_ids]
        if not new_tasks:
            console.print("[yellow]All planned tasks already completed.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in new_tasks:
            if task not in self.tasks:
                self.tasks.append(task)

        # Execute phase - respect dependencies
        await self._execute_tasks(new_tasks)

        # Test phase
        with Status("[bold yellow]Testing...", console=console) as _:
            test_result = await self.tester.run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            
            # Update goal to focus on fixing failures, but preserve original goal
            self.goal = f"Original goal: {self.original_goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        # Build dependency graph
        task_map = {t.id: t for t in self.tasks}
        
        # Topological sort to respect dependencies
        ready_tasks = self._get_ready_tasks(tasks, task_map)
        
        while ready_tasks:
            # Run ready tasks in parallel, respecting max_parallel_tasks
            batch = ready_tasks[: self.settings.max_parallel_tasks]
            ready_tasks = ready_tasks[self.settings.max_parallel_tasks :]
            
            console.print(f"[cyan]Executing {len(batch)} task(s) in parallel...[/cyan]")
            
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in batch
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    task.retry_count += 1
                    
                    if task.retry_count < self.settings.max_task_retries:
                        console.print(f"[yellow]Task {task.id} failed, retrying ({task.retry_count}/{self.settings.max_task_retries})...[/yellow]")
                        logger.warning("task_failed_retry", task_id=task.id, retry=task.retry_count, max_retries=self.settings.max_task_retries, error=str(result))
                        # Re-add to ready tasks for retry
                        ready_tasks.insert(0, task)
                    else:
                        console.print(f"[red]Task {task.id} failed after {task.retry_count} retries: {result}[/red]")
                        logger.error("task_failed_max_retries", task_id=task.id, retries=task.retry_count, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    self._completed_task_ids.add(task.id)
                    self.task_results[task.id] = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)

            # Check for newly ready tasks after this batch completes
            ready_tasks.extend(self._get_newly_ready_tasks(tasks, task_map, ready_tasks))

    def _get_ready_tasks(self, tasks: list[TaskModel], task_map: dict[str, TaskModel]) -> list[TaskModel]:
        """Get tasks that have no unmet dependencies."""
        ready = []
        for task in tasks:
            if task.id in self._completed_task_ids:
                continue
            deps_met = all(
                task_map.get(dep_id, TaskModel(id=dep_id, description="")).status == "completed"
                for dep_id in task.dependencies
            )
            if deps_met:
                ready.append(task)
        return ready

    def _get_newly_ready_tasks(
        self, tasks: list[TaskModel], task_map: dict[str, TaskModel], already_ready: list[TaskModel]
    ) -> list[TaskModel]:
        """Find tasks that became ready after the last batch completed."""
        ready_ids = {t.id for t in already_ready}
        completed_ids = self._completed_task_ids
        newly_ready = []
        
        for task in tasks:
            if task.id in ready_ids or task.id in completed_ids:
                continue
            deps_met = all(
                task_map.get(dep_id, TaskModel(id=dep_id, description="")).status == "completed"
                for dep_id in task.dependencies
            )
            if deps_met:
                newly_ready.append(task)
        return newly_ready

    def _is_done(self) -> bool:
        # Check if all tracked tasks are completed
        if not self.tasks:
            return False
            
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        
        if failed > 0:
            # If there are failed tasks, we're not done (will retry in next cycle)
            return False
            
        return completed >= len(self.tasks)

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status for web UI."""
        return {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "completed_count": len(self._completed_task_ids),
            "total_tasks": len(self.tasks),
            "is_done": self._is_done(),
        }