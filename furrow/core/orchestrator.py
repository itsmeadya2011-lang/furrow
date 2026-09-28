from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        message_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.message_callback = message_callback
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_passed: bool = False
        self._semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                self._save_state(final=False)
                break
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                self._save_state(final=True)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan: Plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = plan.tasks
        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_runs = [self._run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*worker_runs, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._last_test_passed = test_result.passed
        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task: TaskModel) -> Any:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    async def _emit(self, *args: Any, **kwargs: Any) -> None:
        console.print(*args, **kwargs)
        if self.message_callback is not None:
            try:
                buffer = io.StringIO()
                capture = Console(file=buffer, force_terminal=False, no_color=True)
                capture.print(*args, **kwargs)
                await self.message_callback(buffer.getvalue())
            except Exception:  # noqa: BLE001, S110 - best-effort: drop message_callback errors
                pass
    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if not self._last_test_passed:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks

    def _save_state(self, final: bool = False) -> None:
        try:
            state_path = Path(self.client.settings.state_file)
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self._tasks],
                "last_test_passed": self._last_test_passed,
                "complete": final or self._is_done(),
            }
            state_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception:  # noqa: BLE001, S110 - best-effort: state persistence must not raise
            pass
