from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_passed: bool | None = None
        self.state_file = settings.workspace / ".furrow_state.json"
        self._load_state()

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            tests_passed = await self._cycle()
            self._save_state()
            if tests_passed and self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> bool:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.last_test_passed = True
            return True

        # Track new plan tasks in the persistent task history
        self.tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            workers = [WorkerAgent(task=task, client=self.client).run() for task in plan.tasks]
            results = await asyncio.gather(*workers, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_passed = test_result.passed
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
        return test_result.passed

    def _is_done(self) -> bool:
        if not self.tasks:
            return True  # nothing left to do
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed >= len(self.tasks)

    def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test_passed": self.last_test_passed,
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state, indent=2))
        except OSError:
            pass

    def _load_state(self) -> None:
        try:
            if self.state_file.exists():
                data = json.loads(self.state_file.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.last_test_passed = data.get("last_test_passed")
        except (OSError, json.JSONDecodeError, ValueError):
            pass
