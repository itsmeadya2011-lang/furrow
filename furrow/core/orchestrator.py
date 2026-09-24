from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, TaskModel, settings, configure_logging
from furrow.llm import LLMClient

import structlog

configure_logging()

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.state_file = Path(state_file) if state_file else Path.cwd() / settings.state_file
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.tasks))
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug("state_saved", path=str(self.state_file))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)
        while True:
            self.cycles += 1
            
            # Check max_cycles limit
            if settings.max_cycles > 0 and self.cycles > settings.max_cycles:
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Stopping.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                break
                
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_completed", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our task list
        for task in plan.tasks:
            if not any(t.id == task.id for t in self.tasks):
                self.tasks.append(task)

        # Execute pending tasks with retry logic
        await self._execute_tasks_with_retry(plan.tasks)

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
            logger.warning("tests_failed", failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks_with_retry(self, tasks: list[TaskModel], max_retries: int = 2) -> None:
        """Execute tasks with retry logic for failed tasks."""
        tasks_to_run = [t for t in tasks if t.status == "pending"]
        
        for attempt in range(max_retries + 1):
            if not tasks_to_run:
                break
                
            if attempt > 0:
                console.print(f"[yellow]Retry attempt {attempt}/{max_retries} for {len(tasks_to_run)} failed task(s)...[/yellow]")
                logger.info("retry_attempt", attempt=attempt, task_count=len(tasks_to_run))
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                futures = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in tasks_to_run
                ]
                results = await asyncio.gather(*futures, return_exceptions=True)

            # Collect tasks that still need retry
            still_failed = []
            for task, result in zip(tasks_to_run, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("task_failed", task_id=task.id, error=str(result), attempt=attempt)
                    if attempt < max_retries:
                        still_failed.append(task)
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)
            
            tasks_to_run = still_failed

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
            
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        
        # If any task failed, not done (will retry in next cycle)
        if failed > 0:
            return False
            
        # All tasks completed successfully
        if completed >= len(self.tasks):
            return True
            
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
