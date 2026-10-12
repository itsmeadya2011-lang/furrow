from __future__ import annotations

import asyncio
from collections.abc import Callable, Awaitable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TaskStatus, TestResult, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], None] | Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self._on_event = on_event

    async def _emit(self, message: str) -> None:
        if self._on_event is None:
            return
        try:
            result = self._on_event(message)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def run(self) -> None:
        await self._emit("Furrow started")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"Cycle {self.cycles} started")
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as e:
                console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
                await self._emit(f"Cycle {self.cycles} failed: {e}")
                break
            await self._emit(f"Cycle {self.cycles} completed")
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("Goal complete. Halting.")
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({settings.max_cycles}). Halting.[/yellow]")
                await self._emit(f"Reached max_cycles ({settings.max_cycles}). Halting.")
                break

    async def _cycle(self) -> None:
        await self._emit("Planning...")
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan created with {len(plan.tasks)} tasks")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._emit("No tasks planned. Goal may be complete.")
            return

        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
        task_map = {task.id: task for task in plan.tasks}
        completed_ids: set[str] = set()
        failed_ids: set[str] = set()

        while True:
            ready = [
                task
                for task in plan.tasks
                if task.id not in completed_ids
                and task.id not in failed_ids
                and all(dep in completed_ids for dep in task.dependencies)
            ]
            if not ready:
                remaining = [
                    task
                    for task in plan.tasks
                    if task.id not in completed_ids and task.id not in failed_ids
                ]
                if remaining:
                    console.print(
                        f"[red]Dependency cycle or unsatisfiable deps for tasks: "
                        f"{', '.join(t.id for t in remaining)}. Marking failed.[/red]"
                    )
                    await self._emit(
                        f"Dependency cycle for tasks: {', '.join(t.id for t in remaining)}"
                    )
                    for task in remaining:
                        task.status = TaskStatus.FAILED
                        task.result = "Unsatisfiable dependencies"
                        failed_ids.add(task.id)
                break

            async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, Any]:
                async with semaphore:
                    task.status = TaskStatus.IN_PROGRESS
                    await self._emit(f"Task {task.id} started")
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        return task, result
                    except Exception as e:
                        return task, e

            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in ready], return_exceptions=True
            )

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    await self._emit(f"Task {task.id} failed: {result}")
                    failed_ids.add(task.id)
                else:
                    _, task_result = result
                    task.status = TaskStatus.COMPLETED
                    task.result = task_result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    await self._emit(f"Task {task.id} completed")
                    completed_ids.add(task.id)

        await self._emit("Testing...")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            await self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"{self.original_goal}\n\nFix failing tests:\n" + "\n".join(
                test_result.failures
            )
            await self._emit("Will attempt fix in next cycle")

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.plan.tasks if self.plan else []