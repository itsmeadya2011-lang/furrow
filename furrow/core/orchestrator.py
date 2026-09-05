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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.history: list[dict[str, Any]] = []
        self.state_path = state_path or Path(".furrow_state.json")

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if not self.tasks:
                console.print(
                    "[bold green]No tasks remaining. Goal complete. Halting.[/bold green]"
                )
                break
            self._save_state()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.tasks, self.history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        cycle_summary = {
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "results": [(t.id, t.status, t.result) for t in plan.tasks],
            "test_result": test_result.model_dump(),
        }
        self.history.append(cycle_summary)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _save_state(self) -> None:
        try:
            state = {
                "goal": self.goal,
                "cycles": self.cycles,
                "history": self.history,
            }
            self.state_path.write_text(json.dumps(state, indent=2))
        except Exception:
            pass
