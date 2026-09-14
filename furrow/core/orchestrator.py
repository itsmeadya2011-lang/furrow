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
from furrow.config import CycleState, FurrowState, Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, config: Settings | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.config = config or settings
        self.client = client or LLMClient(settings=self.config)
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.state = FurrowState(goal=goal)
        self.resume = resume
        
        setup_logging(self.config.log_level)
        
        if resume and self.config.state_file.exists():
            self.state = FurrowState.load(self.config.state_file)
            self.goal = self.state.goal
            console.print(f"[cyan]Resumed from cycle {self.state.current_cycle}[/cyan]")
            logger.info("resumed_from_state", cycle=self.state.current_cycle, goal=self.goal)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal, resume=self.resume)
        
        start_cycle = self.state.current_cycle + 1 if self.resume else 1
        max_cycles = self.config.max_cycles if self.config.max_cycles > 0 else float('inf')
        
        for cycle in range(start_cycle, max_cycles + 1):
            console.print(f"\n[bold cyan]═══ Cycle {cycle} ═══[/bold cyan]")
            self.state.current_cycle = cycle
            
            await self._cycle()
            
            # Save state after each cycle
            self.state.save(self.config.state_file)
            logger.info("cycle_completed", cycle=cycle, state_file=str(self.config.state_file))
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_completed", total_cycles=cycle)
                break
                
        if cycle >= max_cycles:
            console.print(f"[yellow]Reached max cycles ({max_cycles}). Stopping.[/yellow]")
            logger.warning("max_cycles_reached", max_cycles=max_cycles)

    async def _cycle(self) -> None:
        logger.info("planning_started", cycle=self.state.current_cycle)
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.state.get_all_tasks())
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("planning_completed", cycle=self.state.current_cycle, num_tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.state.current_cycle)
            return

        # Store the plan in state
        self.state.add_cycle(self.state.current_cycle, self.goal, plan)

        logger.info("execution_started", cycle=self.state.current_cycle, num_tasks=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.config).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

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

        logger.info("testing_started", cycle=self.state.current_cycle)
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, plan.tasks)

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

    def _is_done(self) -> bool:
        tasks = self.state.get_all_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False
