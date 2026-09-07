from __future__ import annotations

import asyncio

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


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []  # Track tasks across cycles
        self.max_cycles = self.settings.max_cycles or float('inf')

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            await self._cycle()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            
            if self.cycles >= self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out already-completed tasks from previous cycles
        existing_ids = {t.id for t in self.all_tasks}
        new_tasks = [t for t in plan.tasks if t.id not in existing_ids]
        self.all_tasks.extend(new_tasks)
        
        if not new_tasks:
            console.print("[yellow]All tasks already completed. Re-running tests.[/yellow]")
        else:
            # Respect dependency order: run tasks in dependency order
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                results = await self._execute_tasks(new_tasks)

            for task, result in zip(new_tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[str | Exception]:
        """Execute tasks respecting dependencies."""
        completed_ids: set[str] = set()
        max_parallel = self.settings.max_parallel_tasks or 5
        
        results: list[str | Exception] = [None] * len(tasks)
        task_indices = {t.id: i for i, t in enumerate(tasks)}
        
        async def execute_task(task: TaskModel) -> tuple[str, str | Exception]:
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
                return (task.id, result)
            except Exception as e:
                return (task.id, e)
        
        # Execute tasks in waves based on dependencies
        remaining = list(tasks)
        while remaining:
            # Find tasks whose dependencies are all satisfied
            ready = []
            blocked = []
            for task in remaining:
                deps_satisfied = all(dep in completed_ids for dep in task.dependencies)
                if deps_satisfied:
                    ready.append(task)
                else:
                    blocked.append(task)
            
            if not ready:
                # No tasks ready but still remaining - run all blocked
                if blocked:
                    ready = blocked[:max_parallel]
                    blocked = blocked[max_parallel:]
                else:
                    break
            
            # Limit parallel execution
            ready = ready[:max_parallel]
            
            # Execute ready tasks in parallel
            batch_results = await asyncio.gather(
                *[execute_task(t) for t in ready],
                return_exceptions=True
            )
            
            # Process results
            for task_result in batch_results:
                if isinstance(task_result, tuple):
                    task_id, result = task_result
                    idx = task_indices[task_id]
                    results[idx] = result
                    if not isinstance(result, Exception):
                        completed_ids.add(task_id)
                    remaining = [t for t in remaining if t.id != task_id]
                else:
                    # Exception case
                    remaining = [t for t in remaining if t.id not in completed_ids]
        
        return results

    def _is_done(self) -> bool:
        """Check if all tasks are completed and tests pass."""
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        total = len(self.all_tasks)
        # Done if all tasks completed and none failed
        return completed == total and failed == 0

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
