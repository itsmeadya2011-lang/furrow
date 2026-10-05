from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self._on_event = on_event
        self._stop_requested = False
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.history: list[dict] = []
        self.last_test_result: TestResult | None = None
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()
        self.settings = settings if settings is not None else self.client.settings
        self.planner = PlannerAgent(client=self.client)

    def request_stop(self) -> None:
        self._stop_requested = True

    def _emit(self, event: dict) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:
            # A broken subscriber must never kill the development loop.
            pass

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if not all(t.status == "completed" for t in self.tasks):
            return False
        return self.last_test_result is not None and self.last_test_result.passed

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        try:
            while True:
                if self._stop_requested:
                    self._emit({"type": "done", "reason": "stopped"})
                    console.print("[yellow]Run stopped by request.[/yellow]")
                    return

                if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                    self._emit({"type": "done", "reason": "max_cycles_reached"})
                    console.print("[yellow]Max cycles reached.[/yellow]")
                    break

                self.cycles += 1
                self._emit({"type": "cycle", "cycle": self.cycles})
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

                # Plan
                with Status("[bold yellow]Planning...", console=console):
                    try:
                        plan: Plan = await self.planner.plan(self.goal)
                    except Exception as e:
                        self._emit({"type": "error", "cycle": self.cycles, "error": str(e)})
                        console.print(f"[red]Planning error: {e}[/red]")
                        break

                console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
                self._emit(
                    {
                        "type": "plan",
                        "cycle": self.cycles,
                        "tasks": [t.model_dump() for t in plan.tasks],
                    }
                )

                if not plan.tasks:
                    self._emit({"type": "done", "reason": "no_tasks"})
                    console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                    break

                self.tasks = plan.tasks

                # Execute tasks in parallel with bounded concurrency.
                semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

                async def _run_task(task: TaskModel) -> str:
                    async with semaphore:
                        self._emit(
                            {
                                "type": "task_start",
                                "task_id": task.id,
                                "description": task.description,
                            }
                        )
                        return await WorkerAgent(task=task, client=self.client).run()

                with Status("[bold yellow]Executing tasks in parallel...", console=console):
                    results = await asyncio.gather(
                        *[_run_task(t) for t in self.tasks], return_exceptions=True
                    )

                for task, result in zip(self.tasks, results):
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        console.print(f"[green]Task {task.id} completed[/green]")
                    self._emit(
                        {
                            "type": "task_done",
                            "task_id": task.id,
                            "status": task.status,
                            "result": task.result,
                        }
                    )

                # Test
                with Status("[bold yellow]Testing...", console=console):
                    test_result: TestResult = await TesterAgent(
                        client=self.client
                    ).run(self.goal, self.tasks)
                self.last_test_result = test_result
                self.history.append(
                    {
                        "cycle": self.cycles,
                        "test_summary": test_result.summary,
                        "passed": test_result.passed,
                    }
                )
                self._emit(
                    {
                        "type": "test",
                        "passed": test_result.passed,
                        "summary": test_result.summary,
                        "failures": test_result.failures,
                    }
                )

                if test_result.passed:
                    console.print(f"[green]Tests passed: {test_result.summary}[/green]")
                else:
                    console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                    for failure in test_result.failures:
                        console.print(f"  • {failure}")
                    console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
                    self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
                    continue

                if self._is_done():
                    self._emit({"type": "done", "reason": "goal_complete"})
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
        except asyncio.CancelledError:
            raise
