from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger()

STATE_FILE = Path.cwd() / ".furrow_state.json"


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=settings_obj or settings)
        self.settings = settings_obj or settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.history: list[dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)

        if resume:
            self._load_state()
        else:
            self._save_state()

    async def run(self) -> None:
        log.info("furrow.start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                log.warning("furrow.max_cycles_reached", max_cycles=self.settings.max_cycles)
                console.print("[bold yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/bold yellow]")
                break

            self.cycles += 1
            log.info("furrow.cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                log.info("furrow.goal_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

            self._save_state()

        self._save_state(final=True)
        log.info("furrow.stop", cycles=self.cycles, final=True)

    async def _cycle(self) -> None:
        log.info("furrow.planning", goal=self.goal)
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, existing_tasks=self.tasks)

        log.info("furrow.plan_created", tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            log.warning("furrow.no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.tasks = plan.tasks

        log.info("furrow.executing", task_count=len(plan.tasks))
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings, semaphore=semaphore).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        completed_count = 0
        failed_count = 0
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                failed_count += 1
                log.error("furrow.task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                completed_count += 1
                log.info("furrow.task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        log.info("furrow.execution_complete", completed=completed_count, failed=failed_count)

        log.info("furrow.testing")
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        if test_result.passed:
            log.info("furrow.tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            log.warning("furrow.tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._record_cycle(plan, test_result)

    def _record_cycle(self, plan: Plan, test_result: TestResult) -> None:
        self.history.append(
            {
                "cycle": self.cycles,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "goal": self.goal,
                "tasks": [t.model_dump() for t in plan.tasks],
                "test_result": test_result.model_dump(),
            }
        )

    def _is_done(self) -> bool:
        if not self.tasks:
            return True

        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")

        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _save_state(self, final: bool = False) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "history": self.history,
            "start_time": self.start_time.isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "final": final,
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2))
            log.debug("furrow.state_saved", path=str(STATE_FILE), final=final)
        except Exception as e:
            log.error("furrow.state_save_failed", error=str(e))
            console.print(f"[yellow]Warning: Could not save state: {e}[/yellow]")

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            log.warning("furrow.no_state_file")
            console.print("[yellow]No saved state found, starting fresh.[/yellow]")
            return

        try:
            data = json.loads(STATE_FILE.read_text())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
            self.history = data.get("history", [])
            start_time_str = data.get("start_time")
            if start_time_str:
                self.start_time = datetime.fromisoformat(start_time_str)
            log.info("furrow.state_loaded", cycles=self.cycles, tasks=len(self.tasks))
            console.print(f"[green]Resumed from cycle {self.cycles} with {len(self.tasks)} tasks.[/green]")
        except Exception as e:
            log.error("furrow.state_load_failed", error=str(e))
            console.print(f"[yellow]Warning: Could not load state: {e}[/yellow]")

    def print_summary(self) -> None:
        table = Table(title="Furrow Summary")
        table.add_column("Cycle", style="cyan")
        table.add_column("Tasks", style="magenta")
        table.add_column("Completed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Test Result", style="yellow")

        for h in self.history:
            completed = sum(1 for t in h["tasks"] if t.get("status") == "completed")
            failed = sum(1 for t in h["tasks"] if t.get("status") == "failed")
            test_res = "✓" if h["test_result"].get("passed") else "✗"
            table.add_row(str(h["cycle"]), str(len(h["tasks"])), str(completed), str(failed), test_res)

        console.print(table)