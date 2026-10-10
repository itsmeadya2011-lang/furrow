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
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._stopped = False

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        max_cycles = self.client.settings.max_cycles
        while True:
            if self._stopped:
                console.print("[bold yellow]Stopped by user request.[/bold yellow]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(
                    f"[bold yellow]Max cycles ({max_cycles}) reached. Halting.[/bold yellow]"
                )
                break
            self.cycles += 1
            remaining = len([t for t in self.tasks if t.status in ("pending", "failed")])
            console.print(
                f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]"
                f" [dim]({remaining} tasks remaining)[/dim]"
            )
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            log.warning("Planner returned no tasks", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._merge_tasks(plan.tasks)
        pending = [t for t in self.tasks if t.status in ("pending", "failed")]
        if not pending:
            log.info("No pending tasks to execute", cycle=self.cycles)
            console.print("[yellow]No pending tasks to execute.[/yellow]")
            return

        max_parallel = self.client.settings.max_parallel_tasks
        log.info(
            "Executing tasks",
            cycle=self.cycles,
            pending=len(pending),
            max_parallel=max_parallel,
        )

        with Status(
            f"[bold yellow]Executing tasks (max {max_parallel} parallel)...",
            console=console,
        ):
            for i in range(0, len(pending), max_parallel):
                batch = pending[i : i + max_parallel]
                tasks = [WorkerAgent(task=task, client=self.client).run() for task in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for task, result in zip(batch, results):
                    if isinstance(result, BaseException):
                        task.status = "failed"
                        task.result = str(result)
                        log.error("Task failed", task_id=task.id, error=str(result))
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        log.info("Task completed", task_id=task.id)
                        console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        existing = {t.id: t for t in self.tasks}
        for task in new_tasks:
            if task.id in existing:
                existing[task.id].status = task.status
                existing[task.id].result = task.result
            else:
                self.tasks.append(task)
                existing[task.id] = task

    def _compact_context(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "task_summary": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self.tasks
            ],
            "completed": sum(1 for t in self.tasks if t.status == "completed"),
            "failed": sum(1 for t in self.tasks if t.status == "failed"),
            "pending": sum(1 for t in self.tasks if t.status == "pending"),
        }

    def stop(self) -> None:
        self._stopped = True
        log.info("Stop requested")

    def _is_done(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        pending = sum(1 for t in self.tasks if t.status == "pending")
        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed == len(self.tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks
