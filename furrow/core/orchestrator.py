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
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

configure_logging()
console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, state_file: str | Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.state_file = Path(state_file) if state_file else settings.workspace / ".furrow_state.json"
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                logger.info("state_loaded", state_file=str(self.state_file), cycles=self.cycles)
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        try:
            data = {
                "cycles": self.cycles,
                "goal": self.goal,
                "tasks": [{"id": t.id, "description": t.description, "files": t.files,
                          "dependencies": t.dependencies, "status": t.status, "result": t.result}
                         for t in self.all_tasks],
            }
            self.state_file.write_text(json.dumps(data, indent=2))
            logger.debug("state_saved", state_file=str(self.state_file), tasks=len(self.all_tasks))
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                logger.info("max_cycles_reached", max_cycles=settings.max_cycles)
                console.print(f"[yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/yellow]")
                break
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("orchestrator_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._print_usage()
                self.state_file.unlink(missing_ok=True)
                break

    def _print_usage(self) -> None:
        usage = self.client.usage
        console.print(Panel(
            f"[bold]Token Usage Summary[/bold]\n"
            f"Input tokens: {usage.input_tokens:,}\n"
            f"Output tokens: {usage.output_tokens:,}\n"
            f"Estimated cost: ${usage.total_cost:.4f}",
            title="Usage",
            border_style="magenta"
        ))

    async def _cycle(self) -> None:
        logger.info("planning_start", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("planning_complete", tasks=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.info("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.all_tasks.extend(plan.tasks)

        semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        logger.info("execution_start", tasks=len(plan.tasks), max_parallel=settings.max_parallel_tasks)
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        logger.info("testing_start")
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False
