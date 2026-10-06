from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable

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

STATE_DIR = Path(".furrow")
STATE_FILE = STATE_DIR / "state.json"


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_message: Callable[[str], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.current_tasks: list[Any] = []
        self.on_message = on_message
        self._load_state()

    def _load_state(self) -> None:
        if STATE_FILE.exists():
            try:
                with STATE_FILE.open() as f:
                    state = json.load(f)
                self.goal = state.get("goal", self.goal)
                self.cycles = state.get("cycles", 0)
                self.current_tasks = state.get("current_tasks", [])
                self._print(f"[bold yellow]Resumed from saved state (cycle {self.cycles})[/bold yellow]")
            except Exception as e:
                self._print(f"[red]Failed to load state: {e}[/red]")

    def _save_state(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "current_tasks": [
                {"id": t.id, "status": t.status, "result": t.result} for t in self.current_tasks
            ],
            "timestamp": datetime.now().isoformat(),
        }
        with STATE_FILE.open("w") as f:
            json.dump(state, f, indent=2)

    def _print(self, *args: Any, **kwargs: Any) -> None:
        """Print to the terminal console and, when an ``on_message`` callback is
        set, forward a plain-text rendering of the same output to it.

        The plain-text rendering is produced with a throwaway capture console so
        that Rich markup (e.g. ``[bold green]``) is consumed and only the visible
        text is forwarded to consumers such as the WebSocket web UI.
        """
        console.print(*args, **kwargs)
        if self.on_message is None:
            return
        buf = StringIO()
        capture = Console(file=buf, color_system=None, width=console.width)
        capture.print(*args, **kwargs)
        text = buf.getvalue().rstrip()
        if text:
            self.on_message(text)

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._print(f"[bold yellow]Reached max_cycles limit ({self.client.settings.max_cycles}). Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.current_tasks = plan.tasks
        for task in self.current_tasks:
            task.status = "pending"
            task.result = None

        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_worker(task: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_worker(task) for task in self.current_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(self.current_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.current_tasks)

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._print(f"  • {failure}")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.current_tasks
