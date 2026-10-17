from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import (
    CycleState,
    OrchestratorState,
    Plan,
    Settings,
    TestResult,
    TaskModel,
    settings,
)
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
        resume: bool = False,
    ) -> None:
        self.settings = settings_obj or settings
        self.goal = goal
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed = False
        self.state_path = Path(self.settings.state_file)

        # Resume from saved state if requested and a state file exists.
        if resume and self.state_path.exists():
            self._load_state()

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        logger.info("orchestrator.start", goal=self.goal)

        while not self.completed:
            # Respect max_cycles limit (0 means unlimited).
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]"
                )
                logger.info("orchestrator.max_cycles_reached", max_cycles=self.settings.max_cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator.complete", cycles=self.cycles)
                self.completed = True
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("orchestrator.plan", task_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execute tasks in dependency-respecting waves.
        await self._execute_plan(plan)

        # Store cycle state for persistence.
        cycle_state = CycleState(
            cycle=self.cycles,
            goal=self.goal,
            plan=plan,
            task_results={t.id: t.result or "" for t in plan.tasks},
        )

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(
                client=self.client, settings=self.settings
            ).run(self.goal, plan.tasks)

        cycle_state.test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("orchestrator.tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("orchestrator.tests_failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_plan(self, plan: Plan) -> None:
        """Execute tasks in dependency-respecting waves.

        Tasks with no unmet dependencies run in parallel. After each wave,
        completed tasks are unlocked for their dependents.
        """
        task_map: dict[str, TaskModel] = {t.id: t for t in plan.tasks}
        pending: dict[str, TaskModel] = dict(task_map)
        completed_ids: set[str] = set()

        wave = 0
        while pending:
            wave += 1
            # Find tasks whose dependencies are all completed.
            ready: list[TaskModel] = [
                t
                for t in pending.values()
                if all(dep in completed_ids for dep in t.dependencies)
            ]

            if not ready:
                # Circular dependency or unresolvable dependency.
                stuck = list(pending.keys())
                console.print(
                    f"[red]Cannot resolve dependencies for tasks: {stuck}[/red]"
                )
                logger.error("orchestrator.dependency_cycle", stuck_tasks=stuck)
                for task_id in stuck:
                    pending[task_id].status = "failed"
                    pending[task_id].result = "Unresolvable dependency cycle"
                self.all_tasks.extend(pending.values())
                return

            console.print(
                f"[bold cyan]Wave {wave}: executing {len(ready)} task(s) in parallel[/bold cyan]"
            )
            logger.debug(
                "orchestrator.wave",
                wave=wave,
                task_ids=[t.id for t in ready],
            )

            tasks = [WorkerAgent(task=t, client=self.client, settings=self.settings).run() for t in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(ready, results):
                pending.pop(task.id, None)
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("orchestrator.task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("orchestrator.task_completed", task_id=task.id)
                completed_ids.add(task.id)

        self.all_tasks.extend(plan.tasks)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    def _save_state(self) -> None:
        state = OrchestratorState(
            goal=self.goal,
            cycles=self.cycles,
            completed=self.completed,
        )
        state.cycle_states = []  # Not serializing full cycle states for simplicity
        state.updated_at = datetime.utcnow().isoformat()
        try:
            state.save(self.state_path)
            logger.debug("orchestrator.state_saved", path=str(self.state_path))
        except Exception as e:
            logger.warning("orchestrator.state_save_failed", error=str(e))

    def _load_state(self) -> None:
        try:
            state = OrchestratorState.load(self.state_path)
            self.goal = state.goal or self.goal
            self.cycles = state.cycles
            self.completed = state.completed
            logger.info("orchestrator.state_loaded", cycles=self.cycles, goal=self.goal)
        except Exception as e:
            logger.warning("orchestrator.state_load_failed", error=str(e))
