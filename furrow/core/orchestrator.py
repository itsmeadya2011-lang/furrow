from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.state import State

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._state_file = state_file
        self._no_tasks_streak = 0

        if state_file:
            loaded = State.load(state_file)
            if loaded:
                self.goal = loaded.goal
                self.cycles = loaded.cycles
                self.tasks = loaded.tasks
                self.last_test_result = loaded.last_test_result
                console.print(f"[green]Loaded state from {state_file}[/green]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        completed_summary = self._get_completed_summary()
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, completed_summary)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._no_tasks_streak += 1
            return

        self._no_tasks_streak = 0
        self.tasks = plan.tasks

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_semaphore(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        failed_tasks: list[TaskModel] = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                failed_tasks.append(task)
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        if failed_tasks:
            console.print("[yellow]Spawning fix workers for failed tasks...[/yellow]")
            fix_tasks = []
            for task in failed_tasks:
                fix_task = TaskModel(
                    id=f"fix-{task.id}",
                    description=f"Fix failed task {task.id}: {task.description}\nError: {task.result}",
                    files=task.files,
                    dependencies=[],
                    status="pending",
                )
                fix_tasks.append(fix_task)

            fix_worker_tasks = [run_with_semaphore(task) for task in fix_tasks]
            fix_results = await asyncio.gather(*fix_worker_tasks, return_exceptions=True)

            for task, result in zip(fix_tasks, fix_results):
                if isinstance(result, Exception):
                    console.print(f"[red]Fix task {task.id} failed: {result}[/red]")
                else:
                    console.print(f"[green]Fix task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        if self._state_file:
            State(
                goal=self.goal,
                cycles=self.cycles,
                tasks=self.tasks,
                last_test_result=self.last_test_result,
                last_updated=time.time(),
            ).save(self._state_file)

    def _is_done(self) -> bool:
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True

        if self.last_test_result is not None and not self.last_test_result.passed:
            return False

        if not self.tasks:
            if self._no_tasks_streak >= 2:
                return True
            return False

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")

        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_completed_summary(self) -> str:
        if not self.tasks:
            return "No previous tasks."

        completed = [t for t in self.tasks if t.status == "completed"]
        failed = [t for t in self.tasks if t.status == "failed"]

        lines = []
        if completed:
            lines.append("Completed tasks:")
            for t in completed:
                lines.append(f"  • {t.id}: {t.description}")
                if t.result:
                    lines.append(f"    Result: {t.result[:200]}")

        if failed:
            lines.append("Failed tasks:")
            for t in failed:
                lines.append(f"  • {t.id}: {t.description}")
                if t.result:
                    lines.append(f"    Error: {t.result[:200]}")

        return "\n".join(lines) if lines else "No previous tasks."