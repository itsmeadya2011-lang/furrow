from __future__ import annotations

import asyncio
import json
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
from furrow.config import Plan, Settings, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        console: Console | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self.goal = goal
        self.client = client or LLMClient(settings=self.settings)
        if model is not None:
            self.settings = self.settings.model_copy(update={"model": model})
            self.client = LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = self.settings.max_cycles
        self._current_tasks: list[Any] = []
        self._last_plan_rationale: str = ""
        self._state_file: Path = self.settings.state_file
        self.console = console or Console()
        self._load_state()

    async def run(self) -> None:
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("orchestrator_complete", cycles=self.cycles)
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console) as status:
            plan = await self.planner.plan(self.goal)
        self._last_plan_rationale = plan.rationale
        self._current_tasks = plan.tasks
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._save_state()
            return

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
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

        self._save_state()

    def _is_done(self) -> bool:
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return True
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self._current_tasks

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("state_load_failed", state_file=str(self._state_file), error=str(e))
            return
        self.goal = data.get("goal", self.goal)
        self.cycles = data.get("cycles", 0)
        logger.info("state_restored", state_file=str(self._state_file), cycles=self.cycles)

    def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "last_plan_rationale": self._last_plan_rationale,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self._current_tasks
            ],
        }
        try:
            self._state_file.write_text(json.dumps(state, indent=2))
            logger.info("state_saved", state_file=str(self._state_file), cycles=self.cycles)
        except OSError as e:
            logger.warning("state_save_failed", state_file=str(self._state_file), error=str(e))
