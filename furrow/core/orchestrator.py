from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, settings
from furrow.llm import LLMClient

logger = structlog.get_logger()
console = Console()

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=config)
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.tasks: list[Any] = []
        self.state_path = state_path or (self.config.workspace / self.config.state_file)
        self._load_state()

    def _load_state(self) -> None:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                self.cycles = state.get("cycles", 0)
                self.tasks = state.get("tasks", [])
                logger.info("State loaded", cycles=self.cycles, tasks=len(self.tasks))
            except OSError as e:
                logger.warning("Failed to load state", error=str(e))

    def _save_state(self) -> None:
        try:
            state = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "files": t.files,
                        "dependencies": t.dependencies,
                        "status": t.status,
                        "result": t.result,
                    }
                    for t in self.tasks
                ],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save state", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.config.max_cycles if self.config.max_cycles > 0 else float("inf")

        while self.cycles < max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await asyncio.wait_for(self._cycle(), timeout=self.config.cycle_timeout)
            except asyncio.TimeoutError:
                console.print(f"[red]Cycle timed out after {self.config.cycle_timeout}s[/red]")
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

        if self.cycles >= max_cycles:
            console.print(f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]")

    async def _cycle(self) -> None:
        # Filter pending/in-progress tasks from previous cycles
        pending_tasks = [t for t in self.tasks if t.status in ("pending", "in_progress", "failed")]

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, pending_tasks)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing, avoiding duplicates
        new_task_ids = {t.id for t in plan.tasks}
        self.tasks = [t for t in self.tasks if t.id not in new_task_ids]
        self.tasks.extend(plan.tasks)

        # Execute tasks in parallel with semaphore for max_parallel_tasks
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def run_with_semaphore(task: Any) -> tuple[Any, Any]:
            async with semaphore:
                task.status = "in_progress"
                self._save_state()
                try:
                    result = await WorkerAgent(task=task, client=self.client, settings=self.config).run()
                    return task, result
                except Exception as e:  # noqa: BLE001
                    return task, e

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            task_futures = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*task_futures, return_exceptions=True)

        for item in results:
            if isinstance(item, BaseException):
                raise item
            task, result = item
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        self._save_state()

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[Any]:
        return self.tasks