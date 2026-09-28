from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings
from furrow.core.state import State
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the Furrow development loop.

    Each cycle: plan -> execute (parallel workers) -> test -> persist state.
    The loop halts when every task in the latest plan completed without failure
    *and* tests pass, or when ``max_cycles`` is exhausted.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.settings = Settings()
        self.state = State(path=state_path or Path(".furrow/state.json"))
        self.cycles = 0
        self._plan: Plan | None = None
        self._last_test_passed: bool = False

    async def run(self) -> None:
        # Resume from persisted state if a previous run was interrupted.
        self.state.load()
        if not self.goal and self.state.goal:
            self.goal = self.state.goal
        self.cycles = self.state.cycles

        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )

        while True:
            if (
                self.settings.max_cycles > 0
                and self.cycles >= self.settings.max_cycles
            ):
                console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). "
                    f"Stopping.[/yellow]"
                )
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._plan = plan
        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        # Persist the current goal so it can be resumed.
        self.state.update(goal=self.goal, cycles=self.cycles)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._last_test_passed = True
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

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
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        self._last_test_passed = test_result.passed

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # Persist state for resumption.
        self.state.update(
            cycles=self.cycles, tasks=plan.tasks, last_test_result=test_result
        )

    def _is_done(self) -> bool:
        """Return True when the goal is complete.

        The goal is complete when the latest plan produced no tasks, or when
        every task in the latest plan completed without failure *and* the most
        recent test run passed.
        """
        if self._plan is None or not self._plan.tasks:
            return True

        all_completed = all(t.status == "completed" for t in self._plan.tasks)
        no_failures = not any(t.status == "failed" for t in self._plan.tasks)

        if not (all_completed and no_failures):
            return False

        return self._last_test_passed
