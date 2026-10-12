from __future__ import annotations

import asyncio
import json
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

# Setup logging
setup_logging(log_level=settings.log_level, json_output=settings.log_json)

logger = get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.config = config or settings
        self.client = client or LLMClient(settings=self.config)
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_tasks: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        state_file = self.config.state_file
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("state_loaded", cycles=self.cycles, task_count=len(self.all_tasks))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        state_file = self.config.state_file
        try:
            data = {
                "cycles": self.cycles,
                "goal": self.goal,
                "tasks": [t.model_dump() for t in self.all_tasks],
            }
            with open(state_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("state_saved", cycles=self.cycles, task_count=len(self.all_tasks))
        except OSError as e:
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        logger.info("furrow_started", goal=self.goal, max_cycles=self.config.max_cycles)
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )

        max_cycles = self.config.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                logger.warning("max_cycles_reached", cycles=self.cycles, max_cycles=max_cycles)
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                logger.info("goal_completed", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        logger.info(
            "plan_created", cycle=self.cycles, task_count=len(plan.tasks), rationale=plan.rationale
        )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("no_tasks_planned", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.cycle_tasks = plan.tasks
        self.all_tasks.extend(plan.tasks)

        # Limit parallel tasks
        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def run_task(task: TaskModel):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client, settings=self.config).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = cast(str, result)
                result_str = cast(str, result)
                summary = result_str[:100] if result_str else ""
                logger.info(
                    "task_completed", task_id=task.id, summary=summary
                )
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.config).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks