from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import OrchestratorState, TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("orchestrator")

MAX_RETRIES = 3

OutputCallback = Callable[[str], Any]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: Path | None = None,
        output_callback: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = state_file or settings.state_file
        self._task_retries: dict[str, int] = {}
        self._output_callback = output_callback
        self._load_state()

    def _print(self, message: str) -> None:
        console.print(message)
        if self._output_callback:
            try:
                self._output_callback(message)
            except Exception as e:  # noqa: BLE001
                logger.debug("output_callback_failed", error=str(e))

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                state = OrchestratorState(**data)
                self.goal = state.goal
                self.cycles = state.cycles
                self.all_tasks = state.tasks
                self._print(f"[dim]Loaded state from {self.state_file} (cycle {self.cycles})[/dim]")
                logger.info("state_loaded", state_file=str(self.state_file), cycles=self.cycles)
            except (ValueError, OSError) as e:
                self._print(f"[yellow]Failed to load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        state = OrchestratorState(
            goal=self.goal,
            cycles=self.cycles,
            tasks=self.all_tasks,
            current_cycle=self.cycles,
        )
        self.state_file.write_text(state.model_dump_json(indent=2))
        logger.debug("state_saved", state_file=str(self.state_file), cycles=self.cycles)

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)
        while True:
            # Check max_cycles
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                break

            self.cycles += 1
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", total_cycles=self.cycles)
                self._save_state()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            # Check if task already exists (by id)
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing:
                # Update existing task with new plan info
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
            else:
                self.all_tasks.append(task)

        # Execute tasks with retry logic
        await self._execute_tasks_with_retry(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._print(f"  • {failure}")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks_with_retry(self, tasks: list[TaskModel]) -> None:
        """Execute tasks with retry logic for failures."""
        for task in tasks:
            retries = self._task_retries.get(task.id, 0)
            while retries <= MAX_RETRIES:
                if retries > 0:
                    self._print(f"[yellow]Retrying task {task.id} (attempt {retries}/{MAX_RETRIES})[/yellow]")
                    logger.info("task_retry", task_id=task.id, attempt=retries)

                with Status(f"[bold yellow]Executing task {task.id}...", console=console):
                    result = await WorkerAgent(task=task, client=self.client).run()

                if isinstance(result, Exception) or "error" in result.lower() or "failed" in result.lower():
                    retries += 1
                    self._task_retries[task.id] = retries
                    if retries <= MAX_RETRIES:
                        continue
                    task.status = "failed"
                    task.result = str(result)
                    self._print(f"[red]Task {task.id} failed after {MAX_RETRIES} retries: {result}[/red]")
                    logger.error("task_failed_max_retries", task_id=task.id, error=str(result))
                    break
                else:
                    task.status = "completed"
                    task.result = result
                    self._print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)
                    self._task_retries.pop(task.id, None)
                    break

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
