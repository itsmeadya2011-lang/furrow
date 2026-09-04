from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import aiofiles
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings as default_settings
from furrow.llm import LLMClient

console = Console()
_default_console = console

_logging_level = getattr(logging, default_settings.log_level.upper(), logging.INFO)
logging.basicConfig(level=_logging_level)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(_logging_level),
)
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._semaphore: asyncio.Semaphore | None = None
        self.state_file = self.settings.state_file
        self.console = console or _default_console

    async def run(self) -> None:
        await self._load_state()
        self.console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_start", goal=self.goal, max_cycles=self.settings.max_cycles)
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                logger.info(
                    "max_cycles_reached", cycles=self.cycles, max_cycles=self.settings.max_cycles
                )
                self.console.print(
                    f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]"
                )
                await self._save_state()
                break
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            await self._save_state()
            if self._is_done():
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                await self._save_state()
                break

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        return self._semaphore

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = list(plan.tasks)

        if not self.tasks:
            self.console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        async def run_worker(task: TaskModel) -> tuple[TaskModel, str | Exception]:
            async with self.semaphore:
                logger.info("task_started", task_id=task.id)
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    return task, result
                except Exception as exc:
                    logger.error("task_error", task_id=task.id, error=str(exc))
                    return task, exc

        with Status("[bold yellow]Executing tasks in parallel...", console=self.console):
            results = await asyncio.gather(*[run_worker(t) for t in self.tasks])

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            self.console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning(
                "tests_failed", summary=test_result.summary, failures=test_result.failures
            )
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    async def _save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self.state_file, "w") as f:
                await f.write(json.dumps(state, indent=2, default=str))
            logger.debug("state_saved", path=str(self.state_file))
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("state_save_failed", path=str(self.state_file), error=str(exc))

    async def _load_state(self) -> None:
        if not self.state_file.exists():
            logger.debug("state_file_missing", path=str(self.state_file))
            return
        try:
            async with aiofiles.open(self.state_file, "r") as f:
                content = await f.read()
            data: dict[str, Any] = json.loads(content)
            self.goal = data.get("goal", self.goal)
            self.cycles = int(data.get("cycles", 0))
            self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
            logger.info("state_loaded", path=str(self.state_file), cycles=self.cycles)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("state_load_failed", path=str(self.state_file), error=str(exc))
