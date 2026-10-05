from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
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


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.goal = goal
        self.objective = goal
        self.tasks: list[TaskModel] = []
        self.cycle_history: list[dict] = []
        self.last_test_passed: bool | None = None
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0

    def _emit(self, event: dict) -> None:
        if self.on_event is not None:
            self.on_event(event)

    async def run(self) -> None:
        header = f"[bold green]Furrow[/bold green]\nGoal: {self.goal}"
        console.print(Panel.fit(header, title="Furrow"))
        while True:
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit({"type": "halted", "reason": "max_cycles", "cycles": self.cycles})
                console.print(
                    f"[bold yellow]Reached max_cycles ({max_cycles}). Halting.[/bold yellow]"
                )
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit({"type": "cycle.start", "cycle": self.cycles})
            await self._cycle()
            if self._is_done():
                self._emit({"type": "done", "cycles": self.cycles})
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.objective)
        self.tasks = list(plan.tasks)
        self._emit(
            {"type": "plan", "cycle": self.cycles, "tasks": [t.model_dump() for t in plan.tasks]}
        )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not self.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        for task in self.tasks:
            self._emit({"type": "task.start", "task_id": task.id})

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [WorkerAgent(task=task, client=self.client).run() for task in self.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self._emit({"type": "task.failed", "task_id": task.id, "error": str(result)})
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit({"type": "task.completed", "task_id": task.id, "result": result})
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        self._emit(
            {
                "type": "test.result",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            }
        )

        if test_result.passed:
            self.last_test_passed = True
            self.objective = self.goal
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.last_test_passed = False
            self.objective = "Fix failing tests:\n" + "\n".join(test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

        self.cycle_history.append(
            {
                "cycle": self.cycles,
                "tasks": [t.model_dump() for t in self.tasks],
                "test_passed": self.last_test_passed,
            }
        )
        self._persist_state()

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        if any(t.status == "failed" for t in self.tasks):
            return False
        return all(t.status == "completed" for t in self.tasks) and self.last_test_passed is True

    def _persist_state(self) -> None:
        state_file = self.client.settings.state_file
        if state_file is None:
            return
        payload = {"goal": self.goal, "cycles": self.cycles, "history": self.cycle_history}
        path = Path(state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
