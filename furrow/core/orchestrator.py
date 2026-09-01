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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.console = console or Console()
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.test_result: TestResult | None = None

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit("start", self.goal)
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle", str(self.cycles))
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("done", "Goal complete. Halting.")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit("plan", json.dumps(plan.model_dump()))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit("info", "No tasks planned. Goal may be complete.")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            await self._execute_tasks(plan.tasks)

        for task in plan.tasks:
            if task.status == "completed":
                self.console.print(f"[green]Task {task.id} completed[/green]")
            elif task.status == "failed":
                self.console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

        with Status("[bold yellow]Testing...", console=self.console) as status:
            self.test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if self.test_result.passed:
            self.console.print(f"[green]Tests passed: {self.test_result.summary}[/green]")
            self._emit("test_pass", self.test_result.summary)
        else:
            self.console.print(f"[red]Tests failed: {self.test_result.summary}[/red]")
            for failure in self.test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._emit("test_fail", self.test_result.summary)
            self.goal = f"Fix failing tests:\n" + "\n".join(self.test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        completed_ids: set[str] = set()
        pending = {t.id: t for t in tasks}

        while pending:
            ready = [
                t for t in pending.values()
                if all(dep in completed_ids for dep in t.dependencies)
            ]
            if not ready:
                for t in pending.values():
                    t.status = "failed"
                    t.result = "Dependency not met or circular dependency detected"
                break

            async def run_task(task: Any) -> None:
                async with semaphore:
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        task.status = "completed"
                        task.result = result
                    except Exception as e:
                        task.status = "failed"
                        task.result = str(e)

            batch = ready[:max_parallel]
            await asyncio.gather(*(run_task(t) for t in batch))
            for t in batch:
                completed_ids.add(t.id)
                del pending[t.id]

    def _is_done(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True

        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False

        if completed >= len(tasks):
            if self.test_result is not None and self.test_result.passed:
                return True
            if self.test_result is None:
                return completed > 0

        return False

    def _get_tasks(self) -> list[Any]:
        if self.plan is not None:
            return self.plan.tasks
        return []

    def _emit(self, event: str, message: str) -> None:
        if self.on_event:
            self.on_event(event, message)
