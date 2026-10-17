from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.core.state import STATE_FILE, load_state, save_state

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()
logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, state_file: str | Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: dict[str, Any] = {}
        self.state_file = STATE_FILE if state_file is None else Path(state_file)

    async def run(self) -> None:
        if self.state_file is not None:
            saved = load_state(self.state_file)
            if saved is not None:
                self.goal = saved.get("goal", self.goal)
                self.cycles = saved.get("cycles", 0)
                self.tasks = {t["id"]: TaskModel(**t) for t in saved.get("tasks", [])}
                console.print("[yellow]Resuming from saved state.[/yellow]")

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            max_cycles = self.client.settings.max_cycles
            if max_cycles and self.cycles > max_cycles:
                console.print(f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]")
                break
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

        for task in plan.tasks:
            self.tasks[task.id] = task

        sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_task(task: Any) -> Any:
            async with sem:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    return task, None
                except Exception as exc:
                    task.status = "failed"
                    task.result = str(exc)
                    console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                    return task, exc

        # dependency-aware scheduler
        finished: set[str] = set()
        pending: set[str] = set(t.id for t in plan.tasks)

        # Pre-mark already completed from prior cycles
        for t in plan.tasks:
            if t.status == "completed":
                finished.add(t.id)
                pending.discard(t.id)

        while pending:
            runnable = []
            for tid in list(pending):
                task = self.tasks[tid]
                if all((d in finished) or (self.tasks.get(d) and self.tasks[d].status == "completed") for d in task.dependencies):
                    runnable.append(task)
            if not runnable:
                logger.warning("Circular dependency detected; marking remaining tasks as failed")
                console.print("[yellow]Circular dependency detected; marking remaining tasks as failed[/yellow]")
                for tid in list(pending):
                    self.tasks[tid].status = "failed"
                    self.tasks[tid].result = "circular dependency"
                    finished.add(tid)
                    pending.discard(tid)
                break
            # run runnable in parallel (semaphore limits)
            awaitables = [run_task(t) for t in runnable]
            for tid in runnable:
                pending.discard(tid)
            await asyncio.gather(*awaitables, return_exceptions=True)
            for t in runnable:
                finished.add(t.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        pending_count = sum(1 for t in self._get_tasks() if t.status == "pending")
        console.print(f"[cyan]Cycle summary: completed={completed} failed={failed} pending={pending_count}[/cyan]")

        if self.state_file is not None:
            save_state(self, self.state_file)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return list(self.tasks.values())
