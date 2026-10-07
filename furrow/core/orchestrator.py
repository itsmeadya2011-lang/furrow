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

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        state_file: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient(config or settings)
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = state_file or (self.config.workspace / ".furrow_state.json")
        self._console = console or Console()
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.goal = data.get("goal", self.goal)
                self.original_goal = data.get("original_goal", self.original_goal)
                self._console.print(f"[dim]Loaded state from {self.state_file}[/dim]")
            except Exception as e:
                self._console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "cycles": self.cycles,
                "goal": self.goal,
                "original_goal": self.original_goal,
                "tasks": [t.model_dump() for t in self.all_tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            self._console.print(f"[yellow]Could not save state: {e}[/yellow]")

    async def run(self) -> None:
        self._console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.config.max_cycles and self.cycles >= self.config.max_cycles:
                self._console.print(f"[bold yellow]Reached max cycles ({self.config.max_cycles}). Halting.[/bold yellow]")
                break

            self.cycles += 1
            self._console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if not self.config.max_cycles and self._is_done():
                self._console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self._console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        self._console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracked list
        for task in plan.tasks:
            self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=self._console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.config).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self._console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, self.all_tasks)

        if test_result.passed:
            self._console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            # Restore original goal if we were fixing tests
            if self.goal != self.original_goal:
                self.goal = self.original_goal
        else:
            self._console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._console.print(f"  • {failure}")
            self._console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _cleanup_state(self) -> None:
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception:
                pass
