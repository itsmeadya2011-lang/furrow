from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: Optional[Callable[[str, Any], None]] = None,
        state_path: Path = Path(".furrow_state.json"),
    ) -> None:
        self.original_goal = goal
        self.working_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._all_tasks: list[TaskModel] = []
        self._last_test_result: Optional[TestResult] = None
        self._on_output = on_output
        self.state_path = state_path

    def _print(self, message: str, renderable: Any = None) -> None:
        if self._on_output:
            self._on_output(message, renderable)
        else:
            console.print(message)

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        self._load_state()
        logger.info("run_started", goal=self.original_goal, state_path=str(self.state_path))
        while True:
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                self._print(f"[bold red]Max cycles ({self.client.settings.max_cycles}) reached. Halting.[/bold red]")
                break

            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            logger.debug("state_saved_after_cycle", cycle=self.cycles)

            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        try:
            logger.debug("planning_started", cycle=self.cycles)
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.working_goal, history=self._all_tasks)
            logger.info("planning_complete", cycle=self.cycles, task_count=len(plan.tasks))

            self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                logger.debug("no_tasks_planned", cycle=self.cycles)
                return

            self._all_tasks.extend(plan.tasks)

            logger.debug("execution_started", cycle=self.cycles)
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in plan.tasks
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("execution_complete", cycle=self.cycles)

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.warning("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    self._print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", cycle=self.cycles, task_id=task.id)

            logger.debug("testing_started", cycle=self.cycles)
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await TesterAgent(client=self.client).run(self.working_goal, plan.tasks)
            logger.info("testing_complete", cycle=self.cycles, passed=test_result.passed)

            self._last_test_result = test_result

            if test_result.passed:
                self._print(f"[green]Tests passed: {test_result.summary}[/green]")
                logger.info("tests_passed", cycle=self.cycles)
            else:
                self._print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    self._print(f"  • {failure}")
                self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
                self.working_goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)
                logger.warning("tests_failed", cycle=self.cycles, failure_count=len(test_result.failures))
        except Exception as e:
            logger.exception("cycle_unexpected_error", cycle=self.cycles)
            return

    def _is_done(self) -> bool:
        logger.debug("done_check", cycle=self.cycles)
        if not self._all_tasks:
            return False

        completed = sum(1 for t in self._all_tasks if t.status == "completed")
        failed = sum(1 for t in self._all_tasks if t.status == "failed")

        if failed > 0:
            logger.debug("done_check_result", done=False, reason="has_failures")
            return False

        if completed >= len(self._all_tasks) and self._last_test_result and self._last_test_result.passed:
            logger.debug("done_check_result", done=True)
            return True

        logger.debug("done_check_result", done=False, reason="not_all_completed_or_tests_failed")
        return False

    def _save_state(self) -> None:
        try:
            state = {
                "original_goal": self.original_goal,
                "working_goal": self.working_goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self._all_tasks],
                "last_test_result": self._last_test_result.model_dump() if self._last_test_result else None,
            }
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info("state_saved", path=str(self.state_path))
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))

    def _load_state(self) -> None:
        try:
            if self.state_path.exists():
                with open(self.state_path) as f:
                    state = json.load(f)
                self.original_goal = state.get("original_goal", self.original_goal)
                self.working_goal = state.get("working_goal", self.working_goal)
                self.cycles = state.get("cycles", self.cycles)
                tasks = state.get("tasks", [])
                self._all_tasks = [TaskModel(**t) for t in tasks]
                tr = state.get("last_test_result")
                self._last_test_result = TestResult(**tr) if tr else None
                logger.info("state_loaded", path=str(self.state_path), cycles=self.cycles)
        except Exception as e:
            logger.warning("state_load_failed", error=str(e))

    def save_state(self, path: str | Path) -> None:
        """Save orchestrator state to a JSON file."""
        old_path = self.state_path
        self.state_path = Path(path)
        self._save_state()
        self.state_path = old_path

    def load_state(self, path: str | Path) -> None:
        """Load orchestrator state from a JSON file."""
        old_path = self.state_path
        self.state_path = Path(path)
        self._load_state()
        self.state_path = old_path