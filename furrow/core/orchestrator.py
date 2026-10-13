from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import State, TaskStatus, settings, get_logger
from furrow.llm import LLMClient

console = Console()
logger = get_logger("orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.state = self._load_or_create_state(resume)

    def _load_or_create_state(self, resume: bool) -> State:
        state_file = settings.state_file
        if resume and state_file.exists():
            state = State.from_file(state_file)
            if state and state.goal == self.goal:
                console.print(f"[green]Resumed state from {state_file}[/green]")
                logger.info("resumed_state", state_file=str(state_file), cycles=state.cycles)
                return state
            elif state:
                console.print(f"[yellow]Goal changed, starting fresh (old goal: {state.goal})[/yellow]")
        return State(goal=self.goal)

    def _save_state(self) -> None:
        self.state.updated_at = __import__("datetime").datetime.utcnow().isoformat()
        self.state.to_file(settings.state_file)
        logger.debug("state_saved", tasks=len(self.state.tasks), cycle=self.state.current_cycle)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("start", goal=self.goal)
        
        max_cycles = settings.max_cycles if settings.max_cycles > 0 else float('inf')
        
        while self.state.current_cycle < max_cycles:
            self.state.current_cycle += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.state.current_cycle} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.state.current_cycle)
            await self._cycle()
            self._save_state()
            
            if self.state.is_goal_complete():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.state.current_cycle)
                break
            
            if self.state.current_cycle >= max_cycles:
                console.print(f"[yellow]Max cycles ({max_cycles}) reached.[/yellow]")
                logger.warning("max_cycles_reached", max_cycles=max_cycles)
                break

    async def _cycle(self) -> None:
        # Get ready tasks (dependencies satisfied)
        ready_tasks = self.state.get_ready_tasks()
        
        if not ready_tasks and not self.state.get_in_progress_tasks():
            # No tasks ready, need to plan
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal, self.state)
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
            logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)
            
            if not plan.tasks:
                console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
                return
            
            # Add new tasks to state
            for task in plan.tasks:
                task.cycle = self.state.current_cycle
            self.state.tasks.extend(plan.tasks)
            self._save_state()
            
            # Get ready tasks again after adding new ones
            ready_tasks = self.state.get_ready_tasks()
        
        if not ready_tasks:
            console.print("[yellow]No ready tasks (waiting for dependencies).[/yellow]")
            logger.info("no_ready_tasks", pending=len(self.state.get_pending_tasks()))
            return

        # Execute ready tasks in parallel (respecting max_parallel_tasks)
        max_parallel = settings.max_parallel_tasks
        console.print(f"[cyan]Executing {len(ready_tasks)} tasks (max parallel: {max_parallel})...[/cyan]")
        logger.info("executing_tasks", count=len(ready_tasks), max_parallel=max_parallel)
        
        # Process in batches
        for i in range(0, len(ready_tasks), max_parallel):
            batch = ready_tasks[i:i + max_parallel]
            with Status(f"[bold yellow]Executing batch {i//max_parallel + 1}...", console=console):
                tasks = [
                    WorkerAgent(task=task, client=self.client).run()
                    for task in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.result = str(result)
                    task.updated_at = __import__("datetime").datetime.utcnow().isoformat()
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.updated_at = __import__("datetime").datetime.utcnow().isoformat()
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id, result=result[:100] if result else "")
            
            self._save_state()

        # Run tests
        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, self.state.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", failures=test_result.failures)
            # Add test failures as new tasks for next cycle
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_tasks(self) -> list[Any]:
        return self.state.tasks
