from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.last_plan: Optional[Plan] = None
        self._broadcast: Optional[Callable[[str], Any]] = None

    def set_broadcast(self, callback: Optional[Callable[[str], Any]]) -> None:
        """Register a callback (e.g. WebSocket sender) for status messages."""
        self._broadcast = callback

    def _emit(self, message: str) -> None:
        """Send a status message to the broadcast callback if set."""
        if self._broadcast is not None:
            try:
                result = self._broadcast(message)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception:
                pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self._emit("Furrow started")
        max_cycles = getattr(self.client.settings, "max_cycles", 0)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(f"Cycle {self.cycles} starting")
            try:
                await self._cycle()
            except Exception as e:
                console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
                self._emit(f"Cycle {self.cycles} failed: {e}")
                break
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("Goal complete")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[bold yellow]Max cycles ({max_cycles}) reached. Halting.[/bold yellow]")
                self._emit("Max cycles reached")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            self._emit("Planning...")
            try:
                plan = await self.planner.plan(self.goal)
            except Exception as e:
                console.print(f"[red]Planning failed: {e}[/red]")
                self._emit(f"Planning failed: {e}")
                raise

        self.last_plan = plan
        self.tasks = plan.tasks
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit("No tasks planned")
            return

        await self._execute_tasks(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            self._emit("Testing...")
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._emit(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            self._emit(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        """Execute tasks respecting dependency ordering via topological layers."""
        self._emit("Executing tasks...")
        completed_ids: set[str] = set()
        remaining = list(tasks)
        total = len(remaining)
        done = 0

        while remaining:
            # Find tasks whose dependencies are all satisfied
            ready = [t for t in remaining if all(d in completed_ids for d in t.dependencies)]
            if not ready:
                # Cycle or unresolvable deps — mark all remaining as failed
                for t in remaining:
                    t.status = "failed"
                    t.result = "Unresolved dependencies"
                    console.print(f"[red]Task {t.id} failed: unresolved dependencies[/red]")
                    self._emit(f"Task {t.id} failed: unresolved dependencies")
                break

            # Execute ready tasks in parallel
            with Status(
                f"[bold yellow]Executing {len(ready)} task(s) in parallel...[/bold yellow]",
                console=console,
            ):
                coros = [WorkerAgent(task=t, client=self.client).run() for t in ready]
                results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    self._emit(f"Task {task.id} failed: {result}")
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    self._emit(f"Task {task.id} completed")
                completed_ids.add(task.id)
                done += 1

            remaining = [t for t in remaining if t.id not in completed_ids]

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

    def _get_tasks(self) -> list[Any]:
        return self.tasks
