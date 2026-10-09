from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._no_tasks_planned = False
        self._last_test_passed = False
        self._sem = asyncio.Semaphore(max(1, self.client.settings.max_parallel_tasks))
        self._on_event = on_event

    def _emit(self, event: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:  # noqa: S110, BLE001 - best-effort listener, never break the loop
            pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit({"type": "cycle", "cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as e:  # noqa: BLE001 - report cycle failure and halt
                self._emit({"type": "error", "error": str(e)})
                console.print(f"[red]Error: {e}[/red]")
                break
            if self._no_tasks_planned or self._last_test_passed:
                self._emit({"type": "done", "reason": "goal complete"})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._emit({"type": "done", "reason": "max_cycles reached"})
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        self._tasks = plan.tasks
        self._emit(
            {
                "type": "plan",
                "tasks": [task.model_dump() for task in plan.tasks],
                "rationale": plan.rationale,
            }
        )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._no_tasks_planned = True
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(
                    {
                        "type": "task_done",
                        "task_id": task.id,
                        "status": "failed",
                        "result": str(result),
                    }
                )
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = str(result)
                self._emit(
                    {
                        "type": "task_done",
                        "task_id": task.id,
                        "status": "completed",
                        "result": str(result),
                    }
                )
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await TesterAgent(client=self.client).run(
                self.goal, self._tasks
            )

        self._last_test_passed = test_result.passed

        self._emit(
            {
                "type": "test",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            self.goal = (
                self.original_goal
                + "\n\nFix these failing tests:\n"
                + "\n".join(test_result.failures)
            )
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    async def _run_worker(self, task: TaskModel) -> Any:
        self._emit({"type": "task_start", "task_id": task.id})
        async with self._sem:
            return await WorkerAgent(task=task, client=self.client).run()
