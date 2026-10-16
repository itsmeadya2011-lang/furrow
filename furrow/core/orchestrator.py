from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.state_path = Path(state_path) if state_path else None
        self.console = console or Console()

        if self.state_path and self.state_path.exists():
            self._load_state()

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self.console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self.state_path:
                self._save_state()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_with_sem(task: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await asyncio.gather(
                *(run_with_sem(task) for task in plan.tasks),
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
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _save_state(self) -> None:
        if not self.state_path:
            return
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() if hasattr(t, "model_dump") else t for t in self.tasks],
        }
        self.state_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text())
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            loaded_tasks = state.get("tasks", [])
            self.tasks = [TaskModel(**t) for t in loaded_tasks]
        except (json.JSONDecodeError, ValueError, TypeError):
            self.console.print("[yellow]Could not load state file. Starting fresh.[/yellow]")
