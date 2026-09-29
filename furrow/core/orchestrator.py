from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import CycleState, Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.console: Console = console or Console()
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._state_file = state_file or self.settings.state_file
        self._state: CycleState = self._load_state()

    def _load_state(self) -> CycleState:
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                return CycleState(**data)
            except (json.JSONDecodeError, ValueError) as e:
                self.console.print(f"[yellow]Warning: could not parse state file: {e}[/yellow]")
        return CycleState(goal=self.goal)

    def _save_state(self) -> None:
        try:
            self._state_file.write_text(self._state.model_dump_json(indent=2))
        except OSError as e:
            self.console.print(f"[yellow]Warning: could not save state file: {e}[/yellow]")

    async def run(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            # Stop if we've hit the cycle limit (0 = unlimited)
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                break

            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan: Plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Persist plan tasks so _is_done() and _get_tasks() work across cycles
        self._tasks = plan.tasks
        self._state.cycle = self.cycles
        self._state.goal = self.goal

        # Build a semaphore to enforce max_parallel_tasks
        max_parallel = min(self.settings.max_parallel_tasks, len(plan.tasks))
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_task(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await asyncio.gather(
                *[_run_task(task) for task in plan.tasks],
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console) as status:
            test_result: TestResult = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._state.completed_goals.append(self.goal)
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._state.test_history.append(test_result)
        self._save_state()

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

    def _get_tasks(self) -> list[TaskModel]:
        """Return tasks from the most recent cycle."""
        return self._tasks
