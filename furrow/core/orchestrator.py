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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.all_tasks: list[TaskModel] = []
        self.history: list[dict[str, Any]] = []

    async def run(self) -> None:
        from furrow.config import settings

        self._load_state()
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(
                    f"[yellow]Reached max cycles ({settings.max_cycles}). Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.history.append(
                {"cycle": self.cycles, "status": "no_tasks", "goal": self.goal}
            )
            return

        for task in plan.tasks:
            if not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

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
            elif isinstance(result, dict):
                if result.get("success"):
                    task.status = "completed"
                    task.result = result.get("summary", "")
                    console.print(f"[green]Task {task.id} completed: {result.get('summary', '')}[/green]")
                else:
                    task.status = "failed"
                    task.result = result.get("summary", str(result))
                    console.print(f"[red]Task {task.id} failed: {result.get('summary', '')}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.history.append(
            {
                "cycle": self.cycles,
                "status": "passed" if test_result.passed else "failed",
                "goal": self.goal,
                "test_summary": test_result.summary,
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.plan.tasks if self.plan else []

    def _save_state(self) -> None:
        from furrow.config import settings

        state_path = Path(settings.workspace) / ".furrow_state.json"
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "plan": self.plan.model_dump() if self.plan else None,
            "all_tasks": [t.model_dump() for t in self.all_tasks],
            "history": self.history,
        }
        state_path.write_text(json.dumps(state, indent=2, default=str))

    def _load_state(self) -> None:
        from furrow.config import settings

        state_path = Path(settings.workspace) / ".furrow_state.json"
        if not state_path.exists():
            return
        try:
            data = json.loads(state_path.read_text())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            if data.get("plan"):
                self.plan = Plan(**data["plan"])
            self.all_tasks = [TaskModel(**t) for t in data.get("all_tasks", [])]
            self.history = data.get("history", [])
        except (json.JSONDecodeError, ValueError):
            pass
