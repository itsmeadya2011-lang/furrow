from __future__ import annotations

import asyncio
import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, get_logger
from furrow.llm import LLMClient

console = Console()

logger = get_logger()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._settings = self.client.settings

    async def run(self) -> None:
        self._load_state()
        logger.info("orchestrator_start", goal=self.goal, cycles=self.cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            max_cycles = self._settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                logger.info("max_cycles_reached", max_cycles=max_cycles)
                console.print(f"[yellow]Reached max_cycles ({max_cycles}). Stopping.[/yellow]")
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        logger.info("plan_generated", tasks=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Store tasks for tracking across cycles (updated even when empty
        # so _is_done() correctly detects a completed goal).
        self._tasks = plan.tasks

        if not plan.tasks:
            logger.info("no_tasks_planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_tasks(plan.tasks)

        for task in plan.tasks:
            if task.status == "completed":
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")
            elif task.status == "failed":
                logger.error("task_failed", task_id=task.id, error=task.result)
                console.print(f"[red]Task {task.id} failed: {task.result}[/red]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", summary=test_result.summary)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> None:
        """Execute tasks in parallel waves, respecting dependencies and concurrency limits."""
        max_parallel = self._settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)
        resolved: set[str] = set()
        failed_ids: set[str] = set()

        async def run_one(task: TaskModel) -> None:
            async with semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    task.status = "completed"
                    task.result = result
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    failed_ids.add(task.id)

        while len(resolved) < len(tasks):
            # Mark tasks blocked by failed dependencies
            for t in tasks:
                if t.id not in resolved and any(dep in failed_ids for dep in t.dependencies):
                    t.status = "failed"
                    t.result = "Blocked by failed dependency"
                    resolved.add(t.id)
                    failed_ids.add(t.id)

            # Find ready tasks (all deps resolved)
            ready = [
                t for t in tasks
                if t.id not in resolved
                and all(dep in resolved for dep in t.dependencies)
            ]

            if not ready:
                # Remaining tasks have unresolved dependencies
                for t in tasks:
                    if t.id not in resolved:
                        t.status = "failed"
                        t.result = "Unresolved dependencies"
                        resolved.add(t.id)
                        failed_ids.add(t.id)
                break

            await asyncio.gather(*[run_one(t) for t in ready])
            for t in ready:
                resolved.add(t.id)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[Any]:
        return self._tasks

    def _save_state(self) -> None:
        try:
            state_file = self._settings.state_file
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self._tasks],
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            logger.debug("state_saved", path=str(state_file))
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))

    def _load_state(self) -> None:
        try:
            state_file = self._settings.state_file
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                self.cycles = state.get("cycles", 0)
                task_data = state.get("tasks", [])
                self._tasks = [TaskModel(**t) for t in task_data]
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self._tasks))
        except Exception as e:
            logger.warning("state_load_failed", error=str(e))
