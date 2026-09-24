from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("furrow.orchestrator")

STATE_FILE = ".furrow_state.json"


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
        on_log: Callable[[dict], Any] | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.config = config or settings
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_tasks: list[TaskModel] = []
        self.on_log = on_log or (lambda msg: None)
        workspace = self.config.workspace if config else self.client.settings.workspace
        self.state_path = workspace / STATE_FILE
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.original_goal = data.get("original_goal", self.goal)
                self.goal = data.get("goal", self.goal)
                tasks_data = data.get("all_tasks", [])
                self.all_tasks = [TaskModel(**t) for t in tasks_data]
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.all_tasks))
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        """Persist current state to disk."""
        try:
            data = {
                "cycles": self.cycles,
                "original_goal": self.original_goal,
                "goal": self.goal,
                "all_tasks": [t.model_dump() for t in self.all_tasks],
            }
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("state_saved", path=str(self.state_path))
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))

    def _emit_log(self, level: str, message: str, **kwargs: Any) -> None:
        """Emit a log message via the callback."""
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }
        self.on_log(log_entry)

    async def run(self) -> None:
        self._emit_log("info", f"Starting Furrow: {self.goal}")
        logger.info("starting_furrow", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            self._emit_log("info", f"Cycle {self.cycles} started")
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._emit_log("success", "Goal complete. Halting.")
                logger.info("goal_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._cleanup_state()
                break
            if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                self._emit_log("warning", f"Max cycles ({self.config.max_cycles}) reached. Halting.")
                logger.warning("max_cycles_reached", max_cycles=self.config.max_cycles)
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._emit_log("info", f"Plan created with {len(plan.tasks)} tasks", rationale=plan.rationale)
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit_log("warning", "No tasks planned. Goal may be complete.")
            logger.warning("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.cycle_tasks = plan.tasks
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.config).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit_log("error", f"Task {task.id} failed", task_id=task.id, error=str(result))
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit_log("success", f"Task {task.id} completed", task_id=task.id)
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit_log("success", f"Tests passed: {test_result.summary}")
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit_log("error", f"Tests failed: {test_result.summary}", failures=test_result.failures)
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            self._emit_log("warning", "Will attempt fix in next cycle.")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Keep original goal but add context about failures
            self.goal = f"{self.original_goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _cleanup_state(self) -> None:
        """Remove state file on successful completion."""
        try:
            if self.state_path.exists():
                self.state_path.unlink()
                logger.info("state_cleaned_up")
        except Exception as e:
            logger.warning("state_cleanup_failed", error=str(e))
