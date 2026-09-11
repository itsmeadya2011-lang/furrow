from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

console = Console()
logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, max_cycles: int = 0) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.all_tasks: list[TaskModel] = []
        self.cycle_plans: list[Plan] = []

    async def run(self) -> None:
        setup_logging(self.client.settings.log_level)
        logger.info("furrow.start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle.start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("furrow.complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                logger.warning("furrow.max_cycles_reached", max_cycles=self.max_cycles)
                console.print(f"[yellow]Reached max cycles ({self.max_cycles}). Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self._get_context())
        logger.info("plan.created", tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("plan.empty")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.cycle_plans.append(plan)
        self.all_tasks.extend(plan.tasks)

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
                logger.error("task.failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task.completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.all_tasks)

        if test_result.passed:
            logger.info("tests.passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests.failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Add test failures as context for next cycle instead of mutating goal
            self.goal = f"{self.goal}\n\nPrevious test failures to fix:\n" + "\n".join(f"- {f}" for f in test_result.failures)

    def _get_context(self) -> str:
        """Build context from previous cycles for the planner."""
        if not self.cycle_plans:
            return ""
        
        context_parts = ["Previous cycles:"]
        for i, plan in enumerate(self.cycle_plans, 1):
            context_parts.append(f"\nCycle {i} plan: {plan.rationale}")
            for task in plan.tasks:
                status = task.status
                result = task.result or ""
                context_parts.append(f"  Task {task.id} ({status}): {task.description}")
                if result:
                    context_parts.append(f"    Result: {result[:200]}")
        
        return "\n".join(context_parts)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0:
            return False
        if completed >= len(self.all_tasks) and pending == 0:
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks