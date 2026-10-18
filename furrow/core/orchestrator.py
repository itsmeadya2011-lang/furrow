from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)
console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self._done = False

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator_start", goal=self.goal, max_cycles=self.client.settings.max_cycles)
        while not self._done:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            log.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("goal_complete", cycles=self.cycles)
                self._done = True
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                console.print(f"[bold yellow]Reached max_cycles={self.client.settings.max_cycles}. Stopping.[/bold yellow]")
                log.info("max_cycles_reached", max_cycles=self.client.settings.max_cycles)
                self._done = True
                break

    async def run_stream(self) -> AsyncIterator[str]:
        """Async generator yielding status messages for WebSocket streaming."""
        yield f"Furrow starting. Goal: {self.goal}\n"
        while not self._done:
            self.cycles += 1
            yield f"\n═══ Cycle {self.cycles} ═══\n"
            await self._cycle_stream()
            if self._is_done():
                yield "[bold green]Goal complete. Halting.[/bold green]\n"
                self._done = True
                break
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                yield f"Reached max_cycles={self.client.settings.max_cycles}. Stopping.\n"
                self._done = True
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("plan_created", num_tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            log.info("no_tasks_planned")
            return

        self.tasks = plan.tasks
        await self._execute_tasks()
        await self._test()

    async def _cycle_stream(self) -> None:
        plan = await self.planner.plan(self.goal)
        yield f"Plan: {plan.rationale}\n"
        for t in plan.tasks:
            yield f"  Task {t.id}: {t.description}\n"

        if not plan.tasks:
            yield "No tasks planned. Goal may be complete.\n"
            return

        self.tasks = plan.tasks
        await self._execute_tasks()
        await self._test()

    async def _execute_tasks(self) -> None:
        ordered = self._topological_sort(self.tasks)
        levels = self._group_by_level(ordered)

        for level in levels:
            with Status(f"[bold yellow]Executing {len(level)} tasks in parallel...[/]", console=console):
                coros = [WorkerAgent(task=task, client=self.client).run() for task in level]
                results = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(level, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    log.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    log.info("task_completed", task_id=task.id)

    async def _test(self) -> None:
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self.tasks):
            return True
        return False

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        task_map = {t.id: t for t in tasks}
        visited: set[str] = set()
        result: list[TaskModel] = []

        def visit(task: TaskModel) -> None:
            if task.id in visited:
                return
            visited.add(task.id)
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    visit(task_map[dep_id])
            result.append(task)

        for task in tasks:
            visit(task)
        return result

    def _group_by_level(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        task_map = {t.id: t for t in tasks}
        levels: list[list[TaskModel]] = []
        done: set[str] = set()

        while len(done) < len(tasks):
            level: list[TaskModel] = []
            for task in tasks:
                if task.id in done:
                    continue
                # Only consider deps satisfied if they were completed in PREVIOUS levels
                deps_satisfied = all(d in done for d in task.dependencies if d in task_map)
                if deps_satisfied:
                    level.append(task)
            if not level:
                # Cycle or missing deps; just include remaining
                level = [t for t in tasks if t.id not in done]
                levels.append(level)
                done.update(t.id for t in level)
                break
            levels.append(level)
            done.update(t.id for t in level)
        return levels