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
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: str | Path | None = None,
        websocket: Any = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.state_file = Path(state_file) if state_file else Path.cwd() / ".furrow_state.json"
        self.websocket = websocket
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                self.all_tasks = [TaskModel(**t) for t in data.get("all_tasks", [])]
                self.cycle_history = data.get("cycle_history", [])
                console.print(f"[dim]Resumed from cycle {self.cycles}[/dim]")
            except Exception:
                pass

    def _save_state(self) -> None:
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "all_tasks": [t.model_dump() for t in self.all_tasks],
            "cycle_history": self.cycle_history,
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def _send_ws(self, message: str) -> None:
        if self.websocket:
            try:
                await self.websocket.send_text(message)
            except Exception:
                pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        await self._send_ws(f"Starting Furrow with goal: {self.goal}")
        
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                await self._send_ws("Max cycles reached. Halting.")
                break
                
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._send_ws(f"\n=== Cycle {self.cycles} ===")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._send_ws("Goal complete!")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks, self.cycle_history)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._send_ws(f"Plan: {plan.rationale}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            await self._send_ws("No tasks planned.")
            return

        # Add new tasks to history
        for task in plan.tasks:
            task.status = "pending"
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
                await self._send_ws(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._send_ws(f"Task {task.id} completed: {result}")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        cycle_record = {
            "cycle": self.cycles,
            "plan": plan.model_dump(),
            "test_result": test_result.model_dump(),
        }
        self.cycle_history.append(cycle_record)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._send_ws(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            await self._send_ws(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal with specific failures to fix
            self.goal = self._build_fix_goal(test_result.failures)

    def _build_fix_goal(self, failures: list[str]) -> str:
        return "Fix the following test failures:\n" + "\n".join(f"- {f}" for f in failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
