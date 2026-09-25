from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, max_cycles: int = 0, max_parallel: int = 5) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.max_parallel = max_parallel
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Stopping.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.cycle_history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking
        self.all_tasks.extend(plan.tasks)

        # Execute tasks respecting dependencies
        results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "results": {task.id: task.result for task in plan.tasks},
        })

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        # Build dependency graph
        task_map = {task.id: task for task in tasks}
        completed_ids: set[str] = set()
        results: dict[str, Any] = {}
        
        # Get all task IDs that this cycle's tasks depend on (from previous cycles)
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_map:
                    # Check if it's in all_tasks from previous cycles
                    for prev_task in self.all_tasks:
                        if prev_task.id == dep_id and prev_task.status == "completed":
                            completed_ids.add(dep_id)
                            break
        
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def run_with_semaphore(task: TaskModel):
            async with semaphore:
                # Wait for dependencies
                while True:
                    deps = task.dependencies
                    if all(dep in completed_ids for dep in deps):
                        break
                    await asyncio.sleep(0.5)
                return await WorkerAgent(task=task, client=self.client).run()
        
        # We need to handle dependencies properly - run tasks in order
        # Simple approach: sort by number of dependencies, run in batches
        remaining = list(tasks)
        task_coroutines = {}
        
        while remaining:
            # Find tasks whose dependencies are met
            ready = []
            still_waiting = []
            for task in remaining:
                if all(dep in completed_ids for dep in task.dependencies):
                    ready.append(task)
                else:
                    still_waiting.append(task)
            
            if not ready and still_waiting:
                # Circular dependency or missing dependency - run anyway
                ready = still_waiting[:1]
                still_waiting = still_waiting[1:]
            
            # Launch ready tasks
            for task in ready:
                task_coroutines[task.id] = run_with_semaphore(task)
            
            if task_coroutines:
                # Wait for at least one to complete
                done, pending = await asyncio.wait(
                    task_coroutines.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                for coro in done:
                    # Find which task this was
                    for task_id, task_coro in list(task_coroutines.items()):
                        if task_coro == coro:
                            result = coro.result()
                            results[task_id] = result
                            completed_ids.add(task_id)
                            del task_coroutines[task_id]
                            break
            
            remaining = still_waiting
        
        # Wait for any remaining
        if task_coroutines:
            for task_id, coro in task_coroutines.items():
                result = await coro
                results[task_id] = result
                completed_ids.add(task_id)
        
        # Return results in original task order
        return [results.get(task.id, Exception("Task not executed")) for task in tasks]

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
