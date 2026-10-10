from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    STATE_FILE = ".furrow_state.json"

    def __init__(self, goal: str, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient(config or settings)
        self.planner = PlannerAgent(client=self.client)
        self.config = config or settings
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        state_path = Path(self.STATE_FILE)
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                tasks_data = data.get("tasks", [])
                self._tasks = [TaskModel(**t) for t in tasks_data]
                logger.info("State loaded", cycles=self.cycles, tasks=len(self._tasks))
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning("Failed to load state", error=str(e))

    def _save_state(self) -> None:
        state_path = Path(self.STATE_FILE)
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self._tasks],
        }
        try:
            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning("Failed to save state", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.config.max_cycles
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("Cycle started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]")
                logger.info("Max cycles reached", max_cycles=max_cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self._tasks = plan.tasks

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
                logger.error("Task failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result  # type: ignore[assignment]
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("Task completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.warning("Tests failed", failures=test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self._get_tasks())

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
