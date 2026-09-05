from __future__ import annotations

import asyncio
import json
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
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_status: Any = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.on_status = on_status

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)
        await self._notify("started", {"goal": self.goal})
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._notify("cycle_start", {"cycle": self.cycles})
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_completed", cycles=self.cycles)
                await self._notify("completed", {"cycles": self.cycles})
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
                logger.info("max_cycles_reached", cycles=self.cycles)
                await self._notify("max_cycles", {"cycles": self.cycles})
                break

    async def _notify(self, event: str, data: dict[str, Any]) -> None:
        if self.on_status is not None:
            try:
                await self.on_status(event, data)
            except Exception:
                pass

    async def _cycle(self) -> None:
        # Planning phase
        try:
            with Status("[bold yellow]Planning...", console=console) as status:
                plan = await self.planner.plan(self.goal)
        except Exception as e:
            console.print(f"[red]Planning failed: {e}[/red]")
            logger.error("planning_failed", error=str(e), exc_info=True)
            await self._notify("planning_failed", {"error": str(e)})
            return

        self.plan = plan
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", tasks=len(plan.tasks), rationale=plan.rationale)
        await self._notify("plan_created", {"tasks": len(plan.tasks), "rationale": plan.rationale})

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            await self._notify("no_tasks", {})
            return

        # Execution phase with dependency-aware scheduling
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            await self._execute_tasks(plan.tasks)

        # Update task statuses and log results
        for task in plan.tasks:
            if task.status == "completed":
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)
                await self._notify("task_completed", {"task_id": task.id})
            elif task.status == "failed":
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")
                logger.error("task_failed", task_id=task.id, error=task.result)
                await self._notify("task_failed", {"task_id": task.id, "error": task.result})

        # Testing phase
        try:
            with Status("[bold yellow]Testing...", console=console) as status:
                test_result = await self.tester.run(self.goal, plan.tasks)
        except Exception as e:
            console.print(f"[red]Testing failed: {e}[/red]")
            logger.error("testing_failed", error=str(e), exc_info=True)
            await self._notify("testing_failed", {"error": str(e)})
            return

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
            await self._notify("tests_passed", {"summary": test_result.summary})
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            await self._notify("tests_failed", {"summary": test_result.summary, "failures": test_result.failures})
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[Any]) -> None:
        """Execute tasks respecting dependencies using staged parallel execution."""
        # Build dependency graph
        task_map = {task.id: task for task in tasks}
        completed_ids: set[str] = set()
        remaining = list(tasks)

        while remaining:
            # Find tasks whose dependencies are all satisfied
            ready = []
            still_pending = []
            for task in remaining:
                deps = getattr(task, "dependencies", []) or []
                if all(dep in completed_ids for dep in deps):
                    ready.append(task)
                else:
                    still_pending.append(task)

            if not ready:
                # Circular dependency or unresolvable - run remaining with a warning
                console.print("[yellow]Warning: unresolvable dependencies detected, running remaining tasks anyway.[/yellow]")
                ready = still_pending
                still_pending = []

            # Execute ready tasks in parallel
            coros = [WorkerAgent(task=task, client=self.client).run() for task in ready]
            results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    logger.error("task_execution_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    completed_ids.add(task.id)

            remaining = still_pending

    def _is_done(self) -> bool:
        if not self.plan or not self.plan.tasks:
            return False
        completed = sum(1 for t in self.plan.tasks if t.status == "completed")
        failed = sum(1 for t in self.plan.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.plan.tasks):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        return self.plan.tasks if self.plan else []
