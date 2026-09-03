from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()


def _configure_logging(level: str) -> None:
    """Configure structlog + stdlib logging once, based on settings.log_level."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.KeyValueRenderer(),
        ],
    )


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        resume: bool = False,
    ) -> None:
        # Lazy import to avoid circulars at module load time
        from furrow.config import settings as _default_settings

        self.settings: Settings = settings if settings is not None else _default_settings
        _configure_logging(self.settings.log_level)
        self.log = structlog.get_logger("furrow.core.orchestrator")

        self.goal = goal
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)

        self.cycles = 0
        self.history: list[dict[str, Any]] = []
        self.last_test_passed = False
        self.last_plan: Plan | None = None
        self.completed_task_ids: set[str] = set()

        self.state_file: Path = self.settings.workspace / ".furrow" / "state.json"

        if resume:
            self._load_state()

    # ----- Public lifecycle -----

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        self.log.info("orchestrator.start", goal=self.goal, resume_state=self.state_file.exists())
        try:
            while True:
                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                self._save_state()
                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    self.log.info("orchestrator.done", cycles=self.cycles)
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            self.log.warning("orchestrator.interrupted", cycles=self.cycles)
            self._save_state()

    # ----- Cycle -----

    async def _cycle(self) -> None:
        # Plan
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.last_plan = plan
        self.log.info(
            "plan.generated",
            num_tasks=len(plan.tasks),
            rationale=plan.rationale,
        )
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Treating as complete.[/yellow]")
            self.log.info("plan.empty")
            self.last_test_passed = True
            return

        # Execute with bounded parallelism
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))

        async def _run_with_sem(task: Any) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status(
            f"[bold yellow]Executing up to {self.settings.max_parallel_tasks} tasks in parallel...",
            console=console,
        ):
            coros = [_run_with_sem(t) for t in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        completed_in_cycle = 0
        failed_in_cycle = 0
        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                failed_in_cycle += 1
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self.log.error("task.failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                completed_in_cycle += 1
                self.completed_task_ids.add(task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
                self.log.info("task.completed", task_id=task.id)

        # Test
        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )
        self.last_test_passed = test_result.passed
        self.log.info(
            "test.run",
            passed=test_result.passed,
            summary=test_result.summary,
            failures=test_result.failures,
        )

        summary = {
            "cycle": self.cycles,
            "num_tasks": len(plan.tasks),
            "completed": completed_in_cycle,
            "failed": failed_in_cycle,
            "tests_passed": test_result.passed,
            "summary": test_result.summary,
            "failures": list(test_result.failures),
        }
        self.history.append(summary)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            # Leave goal as-is; _is_done() will return True and the loop will exit.
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            # Keep looping until tests pass.
            self.last_test_passed = False

    # ----- Status -----

    def _is_done(self) -> bool:
        if self.last_test_passed:
            return True
        if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.last_plan.tasks if self.last_plan else []

    # ----- State persistence -----

    def _save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "last_test_passed": self.last_test_passed,
                "completed_task_ids": sorted(self.completed_task_ids),
                "history": self.history,
            }
            # Synchronous JSON write keeps things simple; called once per cycle.
            self.state_file.write_text(json.dumps(data, indent=2))
            self.log.debug("state.saved", path=str(self.state_file))
        except Exception as e:  # pragma: no cover - best-effort persistence
            self.log.warning("state.save_failed", error=str(e))

    def _load_state(self) -> None:
        if not self.state_file.exists():
            self.log.info("state.no_prior_state", path=str(self.state_file))
            return
        try:
            raw = json.loads(self.state_file.read_text())
            self.goal = raw.get("goal", self.goal)
            self.cycles = int(raw.get("cycles", 0))
            self.last_test_passed = bool(raw.get("last_test_passed", False))
            self.completed_task_ids = set(raw.get("completed_task_ids", []))
            self.history = list(raw.get("history", []))
            console.print(
                f"[green]Resumed from state: cycle {self.cycles}, "
                f"{len(self.completed_task_ids)} tasks completed, "
                f"tests_passed={self.last_test_passed}[/green]"
            )
            self.log.info(
                "state.resumed",
                cycles=self.cycles,
                completed=len(self.completed_task_ids),
                last_test_passed=self.last_test_passed,
            )
        except Exception as e:
            self.log.warning("state.load_failed", error=str(e))