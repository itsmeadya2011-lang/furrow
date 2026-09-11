from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
log = get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._current_plan: Plan | None = None
        self._ws_callback: Any = None  # Optional WebSocket callback for streaming

    def set_ws_callback(self, callback: Any) -> None:
        """Set a callback to stream output to a WebSocket."""
        self._ws_callback = callback

    async def _emit(self, message: str) -> None:
        """Emit a message to console and WebSocket if available."""
        console.print(message)
        if self._ws_callback:
            try:
                await self._ws_callback(message)
            except Exception:
                pass

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._current_plan = plan
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
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
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")
                log.info("task_completed", task_id=task.id, result=result)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", summary=test_result.summary)
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
            log.warning("tests_failed", failures=test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self._current_plan is None:
            return []
        return self._current_plan.tasks

    def save_state(self, path: str | Path = ".furrow_state.json") -> None:
        """Persist current state to a JSON file for recovery."""
        if self._current_plan is None:
            return
        import json
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "plan": self._current_plan.model_dump(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("state_saved", path=str(path), cycles=self.cycles)

    def load_state(self, path: str | Path = ".furrow_state.json") -> bool:
        """Load state from a JSON file. Returns True if successful."""
        import json
        p = Path(path)
        if not p.exists():
            return False
        with open(p) as f:
            data = json.load(f)
        self.goal = data["goal"]
        self.cycles = data["cycles"]
        self._current_plan = Plan(**data["plan"])
        log.info("state_loaded", path=str(path), cycles=self.cycles)
        return True