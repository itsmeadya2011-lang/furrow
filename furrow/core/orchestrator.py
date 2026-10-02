from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging_setup import configure_logging, get_logger

__all__ = ["Orchestrator"]

configure_logging()
logger = get_logger("orchestrator")

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings: Settings = settings
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._stopped = False
        self.progress_callback: Callable[[str], None] | None = progress_callback
        self._load_state()

    def _broadcast(self, msg: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(msg)

    async def run(self) -> None:
        logger.info("run_start", goal=self.goal)
        self._broadcast(json.dumps({"type": "progress", "message": "Starting Furrow run", "cycle": self.cycles}))
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            if self._stopped:
                logger.info("stopped", cycle=self.cycles)
                console.print("[bold yellow]Stop requested. Halting.[/bold yellow]")
                break
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._broadcast(json.dumps({"type": "progress", "message": "Goal complete. Halting.", "cycle": self.cycles}))
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycle=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._broadcast(json.dumps({"type": "progress", "message": "Planning complete", "cycle": self.cycles}))
        logger.info("plan_created", task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            test_result = await self.tester.run(self.goal, plan.tasks)
            self._handle_test_result(test_result)
            return

        self._broadcast(json.dumps({"type": "progress", "message": f"Executing {len(plan.tasks)} tasks", "cycle": self.cycles}))

        sem = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _bounded_run(task: TaskModel) -> str:
            async with sem:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(_bounded_run(task) for task in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        self._broadcast(json.dumps({"type": "progress", "message": "All tasks completed", "cycle": self.cycles}))

        self._broadcast(json.dumps({"type": "progress", "message": "Testing...", "cycle": self.cycles}))
        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, plan.tasks)
        self._handle_test_result(test_result)

    def _handle_test_result(self, test_result: TestResult) -> None:
        self._last_test_result = test_result
        if test_result.passed:
            self._broadcast(json.dumps({"type": "progress", "message": f"Tests passed: {test_result.summary}", "cycle": self.cycles}))
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            self._broadcast(json.dumps({"type": "progress", "message": f"Tests failed: {test_result.summary}", "cycle": self.cycles}))
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        if self._last_test_result is not None and not self._last_test_result.passed:
            return False
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self._tasks

    @property
    def state_file(self) -> Any:
        return self.settings.workspace / ".furrow" / "state.json"

    def _save_state(self) -> None:
        path = self.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self._tasks],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("state_saved", path=str(path))

    def _load_state(self) -> None:
        path = self.state_file
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self._tasks = [TaskModel(**t) for t in data.get("tasks", [])]
            logger.info("state_loaded", path=str(path), cycles=self.cycles)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("state_load_failed", error=str(e))

    def stop(self) -> None:
        self._stopped = True
        logger.info("stop_requested")
