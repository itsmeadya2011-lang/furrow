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
from furrow.config import Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        console: Console | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else None) or default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.console = console or Console()
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._tests_passed = False
        self._semaphore: asyncio.Semaphore | None = None

    async def run(self) -> None:
        self.console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        log.info(
            "orchestrator.start", goal=self.goal, max_cycles=self.settings.max_cycles
        )

        # Semaphore to enforce max_parallel_tasks
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        while True:
            self.cycles += 1

            # Enforce max_cycles (0 = unlimited)
            if (
                self.settings.max_cycles > 0
                and self.cycles > self.settings.max_cycles
            ):
                self.console.print(
                    "[yellow]Max cycles reached. Stopping.[/yellow]"
                )
                log.info("orchestrator.max_cycles_reached", cycles=self.cycles)
                break

            self.console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            log.info("orchestrator.cycle_start", cycle=self.cycles)
            await self._cycle()

            if self._is_done() and (
                self.settings.max_cycles == 0
                or self.cycles >= self.settings.max_cycles
            ):
                self.console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("orchestrator.goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=self.console):
            plan = await self.planner.plan(self.goal)
        self.console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )
        log.info("orchestrator.plan_created", tasks=len(plan.tasks))

        if not plan.tasks:
            self.console.print(
                "[yellow]No tasks planned. Goal may be complete.[/yellow]"
            )
            return

        # Store tasks so _is_done() and progress tracking work
        self._tasks = plan.tasks

        with Status(
            "[bold yellow]Executing tasks in parallel...", console=self.console
        ):
            results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("task.failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self.console.print(f"[green]Task {task.id} completed[/green]")
                log.debug("task.completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=self.console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            self._tests_passed = True
            self.console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
            log.info("orchestrator.tests_passed")
        else:
            self._tests_passed = False
            self.console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self.console.print(f"  • {failure}")
            self.console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            log.warning("orchestrator.tests_failed", summary=test_result.summary)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        """Execute tasks respecting dependencies with a bounded concurrency limit.

        Tasks are launched in dependency order: a task only starts once all of
        its declared dependencies have completed.  All eligible tasks run in
        parallel, but the semaphore limits how many execute concurrently.
        """
        assert self._semaphore is not None
        results: list[Any] = [None] * len(tasks)
        task_map: dict[str, tuple[int, TaskModel]] = {
            t.id: (idx, t) for idx, t in enumerate(tasks)
        }
        completed_ids: set[str] = set()
        pending_ids = set(task_map.keys())

        async def _run_one(task_id: str) -> None:
            idx, task = task_map[task_id]
            async with self._semaphore:
                worker = WorkerAgent(task=task, client=self.client)
                try:
                    results[idx] = await worker.run()
                except Exception as e:  # noqa: BLE001 — capture any worker failure
                    results[idx] = e
                finally:
                    completed_ids.add(task_id)

        while pending_ids:
            ready = {
                tid
                for tid in pending_ids
                if all(
                    dep in completed_ids
                    for dep in task_map[tid][1].dependencies
                )
            }

            if not ready:
                # Circular dependency or unsatisfiable — mark remaining as failed
                for tid in pending_ids:
                    idx, task = task_map[tid]
                    results[idx] = RuntimeError(
                        f"Task '{tid}' has unsatisfied dependencies "
                        f"{task.dependencies} — possible circular reference"
                    )
                log.error("orchestrator.unsatisfied_deps", pending=list(pending_ids))
                break

            pending_ids -= ready
            await asyncio.gather(
                *[_run_one(tid) for tid in ready], return_exceptions=True
            )

        return results

    def _is_done(self) -> bool:
        if not self._tasks:
            return True
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self._tasks) and self._tests_passed
