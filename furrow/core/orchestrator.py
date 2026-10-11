from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)

OutputCallback = Callable[[str], None] | Callable[[str], Coroutine[Any, Any, None]]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: OutputCallback | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = Settings()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.output_callback = output_callback or (lambda msg: None)
        self.tasks: list[TaskModel] = []
        if state_file is None:
            state_file = Path(self.settings.workspace) / ".furrow_state.json"
        self.state_file = Path(state_file)

    async def run(self) -> None:
        resumed = self.load_state()
        if resumed:
            resume_msg = f"Resuming from state file: {self.state_file} (cycles completed: {self.cycles})"
        else:
            resume_msg = f"No state file found at {self.state_file}. Starting fresh."
        console.print(f"[bold cyan]{resume_msg}[/bold cyan]")
        self.output_callback(resume_msg)
        msg = f"Furrow\nGoal: {self.goal}"
        console.print(Panel.fit(f"[bold green]{msg}[/bold green]", title="Furrow"))
        self.output_callback(msg)
        while True:
            self.cycles += 1
            cycle_msg = f"═══ Cycle {self.cycles} ═══"
            console.print(f"\n[bold cyan]{cycle_msg}[/bold cyan]")
            self.output_callback(cycle_msg)
            await self._cycle()
            self.save_state()
            if self._is_done():
                done_msg = "Goal complete. Halting."
                console.print(f"[bold green]{done_msg}[/bold green]")
                self.output_callback(done_msg)
                break
            if self.settings.max_cycles and self.cycles >= self.settings.max_cycles:
                console.print(f"[bold yellow]Reached max_cycles limit ({self.settings.max_cycles}). Halting.[/bold yellow]")
                break

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        console.print(
            Panel(
                f"Total cycles: {self.cycles}\n"
                f"Completed tasks: {completed}\n"
                f"Failed tasks: {failed}",
                title="Furrow Summary",
                border_style="green",
            )
        )

    def save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [
                {
                    "id": getattr(task, "id", None),
                    "description": getattr(task, "description", None),
                    "status": getattr(task, "status", None),
                    "result": getattr(task, "result", None),
                }
                for task in self.tasks
            ],
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to save state to %s: %s", self.state_file, e)

    def load_state(self) -> bool:
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load state from %s: %s", self.state_file, e)
            return False
        if not isinstance(state, dict):
            logger.warning("Invalid state file %s: expected a JSON object", self.state_file)
            return False
        goal = state.get("goal")
        cycles = state.get("cycles")
        tasks_data = state.get("tasks")
        if not isinstance(goal, str) or not isinstance(cycles, int) or not isinstance(tasks_data, list):
            logger.warning("Invalid state file %s: missing or malformed fields", self.state_file)
            return False
        tasks: list[TaskModel] = []
        for task_data in tasks_data:
            if not isinstance(task_data, dict):
                logger.warning("Skipping malformed task entry in %s", self.state_file)
                continue
            tasks.append(
                TaskModel(
                    id=str(task_data.get("id", "")),
                    description=str(task_data.get("description", "")),
                    status=str(task_data.get("status", "pending")),
                    result=task_data.get("result"),
                )
            )
        self.goal = goal
        self.cycles = cycles
        self.tasks = tasks
        return True

    def reset_state(self) -> None:
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except OSError as e:
            logger.warning("Failed to reset state file %s: %s", self.state_file, e)

    async def _cycle(self) -> None:
        self.output_callback("Planning...")
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        plan_msg = f"Plan: {plan.model_dump()}"
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.output_callback(plan_msg)

        self.tasks = plan.tasks

        if not plan.tasks:
            no_tasks_msg = "No tasks planned. Goal may be complete."
            console.print(f"[yellow]{no_tasks_msg}[/yellow]")
            self.output_callback(no_tasks_msg)
            return

        self.output_callback("Executing tasks in parallel...")
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
                fail_msg = f"Task {task.id} failed: {result}"
                console.print(f"[red]{fail_msg}[/red]")
                self.output_callback(fail_msg)
            else:
                task.status = "completed"
                task.result = result  # type: ignore[assignment]
                complete_msg = f"Task {task.id} completed"
                console.print(f"[green]{complete_msg}[/green]")
                self.output_callback(complete_msg)

        self.output_callback("Testing...")
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            test_pass_msg = f"Tests passed: {test_result.summary}"
            console.print(f"[green]{test_pass_msg}[/green]")
            self.output_callback(test_pass_msg)
        else:
            test_fail_msg = f"Tests failed: {test_result.summary}"
            console.print(f"[red]{test_fail_msg}[/red]")
            self.output_callback(test_fail_msg)
            for failure in test_result.failures:
                failure_msg = f"  • {failure}"
                console.print(failure_msg)
                self.output_callback(failure_msg)
            fix_msg = "Will attempt fix in next cycle."
            console.print(f"[yellow]{fix_msg}[/yellow]")
            self.output_callback(fix_msg)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks