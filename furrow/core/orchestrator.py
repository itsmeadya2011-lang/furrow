from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_line: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.on_line = on_line

    async def _emit(self, text: str) -> None:
        console.print(text)
        if self.on_line is not None:
            await self.on_line(_strip_ansi(text))

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.plan = plan
            return

        self.plan = plan
        await self._execute_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        task_map = {task.id: task for task in tasks}
        completed_ids = set()
        pending = set(task_map.keys())
        max_parallel = max(1, self.client.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_one(task_id: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task_map[task_id], client=self.client).run()

        while pending:
            ready = [
                task_id
                for task_id in pending
                if all(dep in completed_ids for dep in task_map[task_id].dependencies)
            ]
            if not ready:
                await self._emit("[red]Dependency cycle or missing dependency detected[/red]")
                for task_id in pending:
                    task_map[task_id].status = "failed"
                    task_map[task_id].result = "Unresolved dependency"
                break

            with Status(f"[bold yellow]Executing {len(ready)} task(s)...", console=console):
                results = await asyncio.gather(*(_run_one(task_id) for task_id in ready), return_exceptions=True)

            for task_id, result in zip(ready, results):
                task = task_map[task_id]
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    await self._emit(f"[green]Task {task.id} completed[/green]")
                completed_ids.add(task_id)

            pending -= set(ready)

    def _is_done(self) -> bool:
        if self.plan is None:
            return False

        tasks = self._get_tasks()
        if not tasks:
            return True

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False

        if completed >= len(tasks):
            return True

        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            asyncio.ensure_future(self._emit(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]"))
            return True

        return False

    def _get_tasks(self) -> list[Any]:
        if self.plan is None:
            return []
        return self.plan.tasks