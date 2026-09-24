from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, settings
from furrow.llm import LLMClient
from furrow.log_config import configure_logging, get_logger

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        orchestrator_settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.settings = orchestrator_settings or settings
        self._state_file = self.settings.state_file
        configure_logging(self.settings.log_level)
        self._load_state()

    def _load_state(self) -> None:
        if self._state_file.exists():
            try:
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                logger.info("state_loaded", state_file=str(self._state_file), cycles=self.cycles)
                console.print(f"[dim]Loaded state from {self._state_file}[/dim]")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "files": t.files,
                        "dependencies": t.dependencies,
                        "status": t.status,
                        "result": t.result,
                    }
                    for t in self.all_tasks
                ],
            }
            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("state_saved", state_file=str(self._state_file))
        except OSError as e:
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=self.settings.max_cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_completed", cycles=self.cycles)
                # Clean up state file on completion
                if self._state_file.exists():
                    self._state_file.unlink()
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as _:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Limit parallel tasks
        max_parallel = self.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_worker(task):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_worker(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        # Track all tasks for completion checking
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Testing...", console=console) as _:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)
