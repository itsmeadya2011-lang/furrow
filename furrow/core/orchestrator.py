from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.status_callback = status_callback
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._strip_console = Console(
            record=True, no_color=True, force_terminal=False, width=120
        )

    def _log(self, message: str) -> None:
        if self.status_callback is not None:
            self._strip_console.print(message)
            plain = self._strip_console.export_text()
            self._strip_console.clear()
            self.status_callback(plain)
        else:
            console.print(message)

    async def run(self) -> None:
        self._log(Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
            title="Furrow"
        ))
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles > self.client.settings.max_cycles:
                self._log("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._log(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks
        self._semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        # Execute tasks in parallel (with semaphore limit)
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [self._run_task_with_limit(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._log(f"[green]Task {task.id} completed[/green]")

        # Retry failed tasks
        failed_tasks = [t for t in plan.tasks if t.status == "failed"]
        if failed_tasks:
            self._log(f"[yellow]Retrying {len(failed_tasks)} failed task(s)...[/yellow]")
            retry_results = await asyncio.gather(
                *[self._run_task_with_limit(t) for t in failed_tasks],
                return_exceptions=True,
            )
            for task, result in zip(failed_tasks, retry_results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._log(f"[red]Task {task.id} failed again: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    self._log(f"[green]Task {task.id} completed on retry[/green]")

        # Test
        with Status("[bold yellow]Testing...", console=console):
            self._last_test_result = await TesterAgent(
                client=self.client
            ).run(self.goal, plan.tasks)

        if self._last_test_result.passed:
            self._log(
                f"[green]Tests passed: {self._last_test_result.summary}[/green]"
            )
        else:
            self._log(f"[red]Tests failed: {self._last_test_result.summary}[/red]")
            for failure in self._last_test_result.failures:
                self._log(f"  • {failure}")
            self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            failure_details = "\n".join(
                f"  • {f}" for f in self._last_test_result.failures
            )
            self.goal = f"Fix failing tests:\n{failure_details}"

        self._save_state()

    async def _run_task_with_limit(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        all_completed = all(t.status == "completed" for t in self.tasks)
        if not all_completed:
            return False
        return self._last_test_result is not None and self._last_test_result.passed

    def _get_tasks(self) -> list[Any]:
        return self.tasks

    def _save_state(self) -> None:
        state = {
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self.tasks],
            "test_result": (
                self._last_test_result.model_dump()
                if self._last_test_result
                else None
            ),
        }
        state_file = Path(self.client.settings.workspace) / ".furrow_state.json"
        with contextlib.suppress(OSError):
            state_file.write_text(json.dumps(state, indent=2, default=str))
