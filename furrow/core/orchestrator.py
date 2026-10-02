from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult, settings as default_settings
from furrow.core.state import STATE_FILENAME, StateStore, state_from_dict
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_path: Path | str | None = None,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self._done = False
        self.state_store = StateStore(state_path or STATE_FILENAME)
        if resume:
            self._load_state()

    # -- state persistence ------------------------------------------------
    def _state_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test_result": self.last_test_result.model_dump() if self.last_test_result else None,
            "done": self._done,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _load_state(self) -> None:
        if not self.state_store.path.exists():
            return
        normalized = state_from_dict(self.state_store.load())
        self.goal = normalized["goal"] or self.goal
        self.cycles = normalized["cycles"]
        self.tasks = normalized["tasks"]
        self.last_test_result = normalized["last_test_result"]
        self._done = normalized["done"]
        console.print(f"[dim]Resumed from state file at cycle {self.cycles}.[/dim]")

    def _save_state(self) -> None:
        self.state_store.save(self._state_dict())

    # -- main loop --------------------------------------------------------
    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if self._done:
            console.print("[green]Resumed state indicates goal already complete. Halting.[/green]")
            self._save_state()
            return
        self._save_state()
        while not self._done:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._done = True
                self._save_state()
                break
            if self._max_cycles_reached():
                console.print(f"[yellow]Reached max_cycles={self.settings.max_cycles}. Halting.[/yellow]")
                self._done = True
                self._save_state()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_result = await TesterAgent(client=self.client).run(self.goal, [])
            self._report_test_result(self.last_test_result)
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_runs = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_runs, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._report_test_result(self.last_test_result)

        if not self.last_test_result.passed:
            failures = self.last_test_result.failures or [self.last_test_result.summary]
            self.goal = "Fix failing tests:\n" + "\n".join(f"- {f}" for f in failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    def _report_test_result(self, result: TestResult) -> None:
        if result.passed:
            console.print(f"[green]Tests passed: {result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {result.summary}[/red]")
            for failure in result.failures:
                console.print(f"  • {failure}")

    # -- completion logic -------------------------------------------------
    def _is_done(self) -> bool:
        # No tasks returned by the planner means the goal is considered complete.
        if not self.tasks:
            return True
        if self.last_test_result is None:
            return False
        return self.last_test_result.passed

    def _max_cycles_reached(self) -> bool:
        return self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
