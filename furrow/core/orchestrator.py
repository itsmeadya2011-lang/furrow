from __future__ import annotations

import asyncio
import json
from typing import Any, TextIO, cast

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient


class QueueWriter:
    def __init__(self, queue: asyncio.Queue[str]):
        self.queue = queue

    def write(self, message: str) -> int:
        if message:
            self.queue.put_nowait(message)
        return len(message) if message else 0

    def flush(self) -> None:
        pass

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return 1


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_queue: asyncio.Queue[str] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[Any] = []
        self._last_test_result: TestResult | None = None
        self.output_queue = output_queue
        if output_queue is not None:
            self.console = Console(file=cast(TextIO, QueueWriter(output_queue)))
        else:
            self.console = Console()

    async def run(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"
            )
        )
        while True:
            self.cycles += 1
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self.console.print(
                    f"[bold red]Max cycles ({self.cycles}) reached. Halting.[/bold red]"
                )
                break
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        for task in plan.tasks:
            existing = next((t for t in self._tasks if t.id == task.id), None)
            if existing is not None:
                existing.status = task.status
                existing.result = task.result
            else:
                self._tasks.append(task)

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
                task.result = cast(str, result)
                self.console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._last_test_result = test_result

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
        self.save_state()

    def save_state(self) -> None:
        """Save current state to the configured state file."""
        state = {
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self._tasks],
            "cycles": self.cycles,
        }
        with open(self.client.settings.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> bool:
        """Load state from file. Returns True if state was loaded."""
        path = self.client.settings.state_file
        if not path.exists():
            return False
        try:
            with open(path) as f:
                state = json.load(f)
            self.goal = state.get("goal", self.goal)
            self._tasks = [TaskModel(**t) for t in state.get("tasks", [])]
            self.cycles = state.get("cycles", 0)
            return True
        except (json.JSONDecodeError, OSError, ValueError):
            return False

    def _is_done(self) -> bool:
        if not all(t.status == "completed" for t in self._tasks):
            return False
        return self._last_test_result is not None and self._last_test_result.passed

    def _get_tasks(self) -> list[Any]:
        return self._tasks
