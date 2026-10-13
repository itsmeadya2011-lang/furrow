from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import CycleRecord, OrchestratorState, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()
log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        resume: bool = False,
    ) -> None:
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles or self.client.settings.max_cycles
        self.all_tasks: list[TaskModel] = []
        self.cycle_history: list[dict[str, Any]] = []
        self.goal = goal
        self.state_file = self.client.settings.state_file

        if resume:
            self._load_state()

    def _load_state(self) -> None:
        """Load orchestrator state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                state = OrchestratorState(**data)
                self.goal = state.goal
                self.cycles = state.cycles
                self.all_tasks = state.all_tasks
                self.cycle_history = [record.model_dump() for record in state.cycle_history]
                console.print(f"[green]Resumed from cycle {self.cycles}[/green]")
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.all_tasks))
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]Could not load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        """Save orchestrator state to file."""
        try:
            state = OrchestratorState(
                goal=self.goal,
                cycles=self.cycles,
                all_tasks=self.all_tasks,
                cycle_history=[CycleRecord(**r) for r in self.cycle_history],
            )
            with open(self.state_file, "w") as f:
                json.dump(state.model_dump(), f, indent=2, default=str)
            logger.debug("state_saved", cycles=self.cycles)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", cycles=self.cycles)
                # Clean up state file on completion
                if self.state_file.exists():
                    self.state_file.unlink()
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                logger.info("max_cycles_reached", max_cycles=self.max_cycles)
                break

    async def _cycle(self) -> None:
        # Get codebase context for planning
        codebase_context = self._get_codebase_context()

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, codebase_context)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our tracking
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, codebase_context=codebase_context).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        cycle_results = []
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", cycle=self.cycles, task_id=task.id, error=str(result))
                cycle_results.append({"task_id": task.id, "status": "failed", "error": str(result)})
            else:
                task.status = "completed"
                result_str = str(result)
                task.result = result_str
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", cycle=self.cycles, task_id=task.id)
                cycle_results.append({"task_id": task.id, "status": "completed", "result": result_str})

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            logger.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Update goal with test failures for next cycle
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        # Record cycle history
        self.cycle_history.append({
            "cycle": self.cycles,
            "goal": self.goal,
            "plan": plan.model_dump(),
            "results": cycle_results,
            "test_result": test_result.model_dump()
        })

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        return failed == 0 and completed >= len(self.all_tasks)

    def _get_codebase_context(self) -> str:
        """Get a summary of the codebase for planning context."""
        files = self.client.list_files(self.client.settings.workspace)
        if not files:
            return "Empty workspace."

        # Limit to first 50 files and show structure
        context = f"Workspace: {self.client.settings.workspace}\nFiles ({len(files)} total):\n"
        for f in sorted(files)[:50]:
            context += f"  {f}\n"
        if len(files) > 50:
            context += f"  ... and {len(files) - 50} more files\n"
        return context
