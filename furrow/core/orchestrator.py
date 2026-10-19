from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings_obj or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Append new tasks to the running task list
        self.all_tasks.extend(plan.tasks)

        # Execute tasks respecting dependencies and max_parallel_tasks
        await self._execute_tasks(plan.tasks)

        for task in plan.tasks:
            if task.status == "failed":
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")
            else:
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks respecting dependencies and max_parallel_tasks concurrency."""
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()
        running: dict[str, asyncio.Task] = {}

        async def run_one(task: TaskModel) -> None:
            async with semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    task.status = "completed"
                    task.result = result
                except Exception as exc:  # noqa: BLE001
                    task.status = "failed"
                    task.result = str(exc)
                    console.print(f"[red]Task {task.id} failed: {exc}[/red]")

        # Dependency-aware loop: keep scheduling ready tasks until none remain
        while len(completed_ids) + len(failed_ids) < len(tasks):
            # Launch all ready tasks that aren't already running
            ready = [
                t
                for t in tasks
                if t.id not in completed_ids
                and t.id not in failed_ids
                and t.id not in running
                and all(dep in completed_ids for dep in (t.dependencies or []))
            ]

            if not ready:
                # Nothing ready to run; wait for at least one running task to finish
                if not running:
                    # Deadlock: remaining tasks have unmet dependencies that can never complete
                    for t in tasks:
                        if t.id not in completed_ids and t.id not in failed_ids:
                            t.status = "failed"
                            t.result = "Unresolvable dependency"
                            failed_ids.add(t.id)
                    break
                done, _ = await asyncio.wait(
                    list(running.values()), return_when=asyncio.FIRST_COMPLETED
                )
                for fut in done:
                    task_id = next(tid for tid, t in running.items() if t is fut)
                    del running[task_id]
                    if fut.exception() is None:
                        completed_ids.add(task_id)
                    else:
                        failed_ids.add(task_id)
                continue

            for task in ready:
                running[task.id] = asyncio.create_task(run_one(task))

            if not running:
                break

            done, _ = await asyncio.wait(
                list(running.values()), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                task_id = next(tid for tid, t in running.items() if t is fut)
                del running[task_id]
                if fut.exception() is None:
                    completed_ids.add(task_id)
                else:
                    failed_ids.add(task_id)

        # Await any remaining running tasks
        if running:
            await asyncio.gather(*running.values(), return_exceptions=True)
            for task_id, fut in running.items():
                if fut.exception() is None:
                    completed_ids.add(task_id)
                else:
                    failed_ids.add(task_id)

    def _is_done(self) -> bool:
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        tasks = self._get_tasks()
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if tasks and all(t.status == "completed" for t in tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.all_tasks
