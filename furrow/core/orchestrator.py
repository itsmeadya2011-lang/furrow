from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient

console = Console()

STATE_FILE = Path(".furrow_state.json")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, max_cycles: int = 0, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict] = []
        
        if resume:
            self._load_state()

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.cycle_history = data.get("cycle_history", [])
                console.print(f"[green]Resumed from state: {self.cycles} cycles, {len(self.all_tasks)} tasks[/green]")
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
                "cycle_history": self.cycle_history,
            }
            STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Could not save state: {e}[/yellow]")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.max_cycles > 0 and self.cycles > self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                break
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                # Clean up state file on completion
                if STATE_FILE.exists():
                    STATE_FILE.unlink()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as _:
            plan = await self.planner.plan(self.goal, self.all_tasks, self.cycle_history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking
        for task in plan.tasks:
            self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_result: dict = {"cycle": self.cycles, "tasks": [], "test_result": None}
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                cycle_result["tasks"].append({"id": task.id, "status": "failed", "result": str(result)})
            else:
                task.status = "completed"
                task.result = result if isinstance(result, str) else str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                cycle_result["tasks"].append({"id": task.id, "status": "completed", "result": result})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        cycle_result["test_result"] = test_result.model_dump()
        self.cycle_history.append(cycle_result)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Prepend failures to goal for next cycle
            self.goal = f"Previous goal: {self.goal}\n\nFix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
