from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import aiofiles
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, TaskModel
from furrow.llm import LLMClient


class Orchestrator:
    """Orchestrates planning, execution, and testing across multiple cycles.

    The orchestrator drives a goal through repeated cycles: the planner
    produces a plan of tasks, workers execute those tasks in parallel, and
    a tester verifies the outcome. Tasks are accumulated across cycles so
    the orchestrator can track overall progress toward the goal.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: str | Path = ".furrow/state.json",
        console: Console | None = None,
    ) -> None:
        """Initialize the orchestrator with a goal and optional LLM client.

        Args:
            goal: The high-level objective to accomplish.
            client: Optional LLM client. A new one is created if omitted.
            state_file: Path to the state file for persistence.
            console: Optional rich Console for output. Creates a new one if omitted.
        """
        self.goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = Settings()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_file = Path(state_file)
        self.console = console or Console()

    async def run(self) -> None:
        """Run cycles until the goal is complete or max cycles is reached."""
        # Try to load existing state
        await self.load_state()

        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            await self.save_state()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self.save_state()
                break

    async def _cycle(self) -> None:
        """Execute a single planning, execution, and testing cycle."""
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Track tasks across cycles so progress accumulates.
        self.tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

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
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Append failures to the existing goal so the planner knows what
            # still needs fixing instead of discarding prior context.
            self.goal = (
                f"{self.goal}\n\nFix the following failing tests:\n"
                + "\n".join(test_result.failures)
            )

    def _is_done(self) -> bool:
        """Determine whether the orchestrator should stop running cycles.

        Returns True if all tracked tasks are completed with no failures,
        or if the maximum number of cycles has been reached (when set).
        """
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True

        if not self.tasks:
            return False

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tasks tracked across cycles.

        Returns:
            The accumulated list of tasks from every cycle so far.
        """
        return self.tasks

    def reset(self, goal: str | None = None) -> None:
        """Reset the orchestrator for a new goal.

        Clears the accumulated tasks and cycle counter. If a new goal is
        provided it replaces the current one.

        Args:
            goal: Optional new goal string. If None, the existing goal is kept.
        """
        self.tasks = []
        self.cycles = 0
        if goal is not None:
            self.goal = goal

    async def save_state(self) -> None:
        """Save current state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        async with aiofiles.open(self.state_file, "w") as f:
            await f.write(json.dumps(state, indent=2))

    async def load_state(self) -> None:
        """Load state from disk if it exists."""
        if not self.state_file.exists():
            return
        try:
            async with aiofiles.open(self.state_file, "r") as f:
                data = json.loads(await f.read())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
            self.console.print(f"[yellow]Resuming from cycle {self.cycles} with {len(self.tasks)} tasks.[/yellow]")
        except Exception as e:
            self.console.print(f"[yellow]Could not load state: {e}[/yellow]")

    def clear_state(self) -> None:
        """Delete the state file."""
        if self.state_file.exists():
            self.state_file.unlink()