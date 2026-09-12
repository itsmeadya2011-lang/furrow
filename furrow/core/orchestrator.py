from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
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

console = Console()
logger = logging.getLogger("furrow")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.cycle_history: list[dict] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"furrow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level))
        logger.info(f"Starting Furrow with goal: {self.goal}")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("Starting Furrow run loop")
        while True:
            self.cycles += 1
            
            # Check max cycles limit
            if settings.max_cycles > 0 and self.cycles > settings.max_cycles:
                console.print(f"[bold yellow]Max cycles ({settings.max_cycles}) reached. Halting.[/bold yellow]")
                logger.warning(f"Max cycles ({settings.max_cycles}) reached")
                break
                
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info(f"Starting cycle {self.cycles}")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("Goal complete, halting")
                break

    async def _cycle(self) -> None:
        # Provide context from previous cycles to planner
        context = self._build_context()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info(f"Cycle {self.cycles} plan: {plan.rationale}")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("No tasks planned")
            return

        # Track tasks for this cycle
        for task in plan.tasks:
            task.status = "pending"
            self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            # Limit parallel tasks
            semaphore = asyncio.Semaphore(settings.max_parallel_tasks)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()
            
            tasks = [run_with_semaphore(task) for task in plan.tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error(f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info(f"Task {task.id} completed: {result}")

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in plan.tasks],
        })

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info(f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning(f"Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                logger.warning(f"  Failure: {failure}")
            # Keep original goal but add fix context
            self.goal = f"{self.original_goal}\n\nPrevious test failures to fix:\n" + "\n".join(test_result.failures)

    def _build_context(self) -> str:
        """Build context from previous cycles for the planner."""
        if not self.cycle_history:
            return ""
        
        context_parts = ["Previous cycles summary:"]
        for cycle in self.cycle_history[-3:]:  # Last 3 cycles
            context_parts.append(f"\nCycle {cycle['cycle']}:")
            context_parts.append(f"  Goal: {cycle['goal'][:200]}...")
            for task in cycle['tasks']:
                context_parts.append(f"  Task {task['id']}: {task['description']} ({task.get('status', 'pending')})")
                if task.get('result'):
                    context_parts.append(f"    Result: {task['result'][:200]}")
        return "\n".join(context_parts)

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
