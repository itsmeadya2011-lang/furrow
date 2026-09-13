from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()
OutputCallback = Callable[[str], Any]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.output_callback = output_callback
        self.logger = logging.getLogger("furrow")
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _print(self, message: str) -> None:
        if self.output_callback:
            self.output_callback(str(message))
        else:
            console.print(message)

    async def run(self) -> None:
        self._print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                self.logger.info("Max cycles (%d) reached. Halting.", self.settings.max_cycles)
                self._print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self._print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._print("[bold green]Goal complete. Halting.[/bold green]")
                self.logger.info("Goal complete after %d cycles.", self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self._print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self.tasks = plan.tasks

        if not plan.tasks:
            self._print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                self.logger.info("Starting task %s: %s", task.id, task.description)
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                result = await worker.run()
                self.logger.info("Completed task %s", task.id)
                return result

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._print(f"[red]Task {task.id} failed: {result}[/red]")
                self.logger.error("Task %s failed: %s", task.id, result)
            else:
                task.status = "completed"
                task.result = result
                self._print(f"[green]Task {task.id} completed[/green]")
                self.logger.info("Task %s completed", task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            self._print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.logger.info("Tests passed: %s", test_result.summary)
        else:
            self._print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._print(f"  • {failure}")
            self._print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.logger.warning("Tests failed: %s", test_result.summary)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False
