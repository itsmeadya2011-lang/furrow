from __future__ import annotations

import asyncio
from collections import defaultdict, deque

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as _:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_tasks_by_dependency(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as _:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks_by_dependency(self, tasks: list[TaskModel]) -> None:
        task_by_id = {task.id: task for task in tasks}
        in_degree = defaultdict(int)
        adj = defaultdict(list)

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_by_id:
                    adj[dep_id].append(task.id)
                    in_degree[task.id] += 1

        queue = deque([task.id for task in tasks if in_degree[task.id] == 0])
        waves: list[list[str]] = []

        while queue:
            wave_size = len(queue)
            wave = []
            for _ in range(wave_size):
                task_id = queue.popleft()
                wave.append(task_id)
                for neighbor in adj[task_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            waves.append(wave)

        executed = set()
        for wave in waves:
            if not wave:
                continue
            wave_tasks = [task_by_id[tid] for tid in wave if tid not in executed]
            if not wave_tasks:
                continue
            with Status(f"[bold yellow]Executing wave of {len(wave_tasks)} tasks...", console=console):
                tasks_to_run = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in wave_tasks
                ]
                results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            for task, result in zip(wave_tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                executed.add(task.id)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        return failed == 0 and completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        if self.plan is None:
            return []
        return self.plan.tasks
