from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.context import ContextCompactor
from furrow.llm import LLMClient
from furrow.state import FurrowState, StateManager, get_state_manager

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        workspace: Path | str = ".",
        state_manager: StateManager | None = None,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.compactor = ContextCompactor(client=self.client)
        self.cycles = 0
        self.workspace = Path(workspace)
        self.state_manager = state_manager or get_state_manager(self.workspace)
        self.resume = resume

    async def run(self) -> None:
        state = self.state_manager.load()

        if self.resume and state.goal:
            self.goal = state.goal
            # Compact context if needed when resuming
            compacted_context = await self.compactor.compact_state(state, self.goal)
            logger.info("orchestrator_resume", goal=self.goal, previous_cycles=len(state.cycles), context_length=len(compacted_context))
            console.print(Panel.fit(
                f"[bold green]Furrow (Resumed)[/bold green]\nGoal: {self.goal}\nPrevious cycles: {len(state.cycles)}",
                title="Furrow"
            ))
        else:
            logger.info("orchestrator_start", goal=self.goal)
            console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            # Check max cycles
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                logger.info("orchestrator_max_cycles_reached", max_cycles=self.client.settings.max_cycles)
                console.print(f"[yellow]Max cycles ({self.client.settings.max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            logger.info("orchestrator_cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("orchestrator_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.debug("cycle_planning_start", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.debug("cycle_planning_done", task_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.state_manager.start_cycle(self.goal, plan.rationale, plan.tasks)

        logger.debug("cycle_execution_start", task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self.state_manager.update_task_result(task.id, "failed", str(result))
                logger.warning("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                self.state_manager.update_task_result(task.id, "completed", result)

        logger.debug("cycle_testing_start")
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.state_manager.set_test_result(
            test_result.passed,
            test_result.summary,
            test_result.failures
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("cycle_tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.warning("cycle_tests_failed", summary=test_result.summary, failures=test_result.failures)

    def _is_done(self) -> bool:
        state = self.state_manager.get_state()
        if not state.cycles:
            return False

        latest = state.cycles[-1]
        if not latest.tasks:
            return True

        completed = sum(1 for t in latest.tasks if t.get("status") == "completed")
        failed = sum(1 for t in latest.tasks if t.get("status") == "failed")

        if failed > 0:
            return False
        return completed >= len(latest.tasks)

    def get_state(self) -> FurrowState:
        return self.state_manager.get_state()