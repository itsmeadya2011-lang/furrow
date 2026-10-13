from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import CycleState, Settings, State
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.state = State.load(self.settings.state_file)
        if not self.state.goal:
            self.state.goal = goal
        self.cycles = self.state.current_cycle
        logger.info("orchestrator_initialized", goal=goal, cycles=self.cycles)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        max_cycles = self.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached. Halting.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=max_cycles)
                break
                
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)
            try:
                await self._cycle()
            except Exception as e:  # noqa: BLE001
                logger.error("cycle_failed", cycle=self.cycles, error=str(e))
                console.print(f"[red]Cycle {self.cycles} failed: {e}[/red]")
            self.state.save(self.settings.state_file)
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self.state)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            cycle_state = CycleState(cycle=self.cycles, goal=self.goal, plan=plan)
            self.state.add_cycle(cycle_state)
            return

        # Filter tasks that are already completed
        pending_tasks = [t for t in plan.tasks if t.status == "pending"]
        if not pending_tasks:
            console.print("[yellow]All tasks already completed.[/yellow]")
            cycle_state = CycleState(cycle=self.cycles, goal=self.goal, plan=plan)
            self.state.add_cycle(cycle_state)
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in pending_tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(pending_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result if isinstance(result, str) else str(result)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        cycle_state = CycleState(
            cycle=self.cycles,
            goal=self.goal,
            plan=plan,
            test_result=test_result,
        )
        self.state.add_cycle(cycle_state)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        all_tasks = self.state.get_all_tasks()
        if not all_tasks:
            return False
            
        completed = sum(1 for t in all_tasks if t.status == "completed")
        failed = sum(1 for t in all_tasks if t.status == "failed")
        pending = sum(1 for t in all_tasks if t.status == "pending")
        
        if failed > 0 and self.settings.max_cycles == 0:
            return False  # Keep trying to fix
        if pending > 0:
            return False
        return bool(completed >= len(all_tasks) and len(all_tasks) > 0)

    def _get_tasks(self) -> list[Any]:
        return self.state.get_all_tasks()
