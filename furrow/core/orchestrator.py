from __future__ import annotations

import asyncio
import json
import os
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
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)
        self._load_state()

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break
            self._save_state()

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.tasks = plan.tasks
        logger.info("plan_generated", tasks=len(self.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            async def _run_task(task: TaskModel) -> TaskModel:
                async with self._semaphore:
                    logger.info("task_started", task_id=task.id)
                    try:
                        result = await WorkerAgent(task=task, client=self.client).run()
                        task.status = "completed"
                        task.result = result
                        console.print(f"[green]Task {task.id} completed[/green]")
                        logger.info("task_completed", task_id=task.id)
                    except Exception as exc:
                        task.status = "failed"
                        task.result = str(exc)
                        console.print(f"[red]Task {task.id} failed: {exc}[/red]")
                        logger.error("task_failed", task_id=task.id, error=str(exc))
                return task

            tasks = [_run_task(task) for task in plan.tasks]
            await asyncio.gather(*tasks, return_exceptions=True)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _save_state(self) -> None:
        state_dir = Path(self.client.settings.workspace) / ".furrow"
        state_dir.mkdir(exist_ok=True)
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [task.model_dump() for task in self.tasks],
        }
        path = state_dir / "state.json"
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info("state_saved", path=str(path))
        except Exception as exc:
            logger.error("state_save_failed", error=str(exc))

    def _load_state(self) -> None:
        path = Path(self.client.settings.workspace) / ".furrow" / "state.json"
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", self.cycles)
            raw_tasks = state.get("tasks", [])
            self.tasks = [TaskModel(**t) for t in raw_tasks]
            logger.info("state_loaded", path=str(path), cycles=self.cycles, tasks=len(self.tasks))
        except Exception as exc:
            logger.error("state_load_failed", error=str(exc))
