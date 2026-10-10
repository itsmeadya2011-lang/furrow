from __future__ import annotations

import asyncio
import json
import logging
import os
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
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger()


def configure_logging(verbose: bool = False) -> None:
    """Configure structured logging to `.furrow/furrow.log`."""
    log_dir = Path(".furrow")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "furrow.log"
    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=open(log_file, "a")),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._failed_task_descriptions: list[str] = []

    @property
    def state_file(self) -> Path:
        return self.client.settings.workspace / ".furrow_state.json"

    def save_state(self) -> None:
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self._tasks],
            "failed_task_descriptions": self._failed_task_descriptions,
        }
        path = self.state_file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, default=str))

    def load_state(self) -> bool:
        path = self.state_file
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"[red]Failed to load state file: {e}[/red]")
            log.error("state_load_failed", error=str(e))
            return False
        self.goal = state.get("goal", self.goal)
        self.cycles = state.get("cycles", 0)
        self._tasks = [TaskModel(**t) for t in state.get("tasks", [])]
        self._failed_task_descriptions = state.get("failed_task_descriptions", [])
        return True

    async def run(self, resume: bool = False) -> None:
        if resume:
            if self.load_state():
                console.print("[green]Resuming from saved state.[/green]")
                log.info("resumed_from_state", cycles=self.cycles)
            else:
                console.print("[yellow]No saved state found. Starting fresh.[/yellow]")
                log.warning("no_saved_state_found")
        else:
            path = self.state_file
            if path.exists():
                path.unlink()
                console.print("[yellow]Cleared existing state file.[/yellow]")
                log.info("cleared_existing_state")
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator_started", goal=self.goal)
        max_cycles = self.client.settings.max_cycles
        while True:
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(f"[bold yellow]Max cycles ({max_cycles}) reached. Halting.[/bold yellow]")
                log.warning("max_cycles_reached", max_cycles=max_cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            log.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            self.save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("goal_complete", cycles=self.cycles)
                break

    def _detect_cycles(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Detect cycles in the dependency graph and break them by removing edges."""
        task_ids = {t.id for t in tasks}
        # Build adjacency list
        graph: dict[str, list[str]] = {t.id: [d for d in t.dependencies if d in task_ids] for t in tasks}

        visited: set[str] = set()
        rec_stack: set[str] = set()
        cycle_nodes: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for dep in graph.get(node, []):
                if dep not in visited:
                    if dfs(dep):
                        cycle_nodes.add(node)
                        return True
                elif dep in rec_stack:
                    cycle_nodes.add(node)
                    cycle_nodes.add(dep)
                    return True
            rec_stack.discard(node)
            return False

        for t in tasks:
            if t.id not in visited:
                dfs(t.id)

        if cycle_nodes:
            console.print(f"[bold red]Warning: dependency cycle detected involving {sorted(cycle_nodes)}. Breaking cycle.[/bold red]")
            for t in tasks:
                if t.id in cycle_nodes:
                    # Drop dependencies that point into the cycle to break it
                    t.dependencies = [d for d in t.dependencies if d not in cycle_nodes]
        return tasks

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Sort tasks so that dependencies come before the tasks that depend on them."""
        task_ids = {t.id for t in tasks}
        # Only consider dependencies that exist within this task set
        graph: dict[str, list[str]] = {t.id: [d for d in t.dependencies if d in task_ids] for t in tasks}
        in_degree = {t.id: len(graph[t.id]) for t in tasks}
        # Tasks with no dependencies come first
        sorted_tasks = [t for t in tasks if in_degree[t.id] == 0]
        remaining = [t for t in tasks if in_degree[t.id] > 0]
        result: list[TaskModel] = []
        # Simple iterative approach: repeatedly pick tasks whose deps are satisfied
        pending = list(tasks)
        while pending:
            ready = [t for t in pending if all(d in {r.id for r in result} or d not in task_ids for d in t.dependencies)]
            if not ready:
                # Should not happen after cycle breaking, but guard against infinite loop
                result.extend(pending)
                break
            for t in ready:
                result.append(t)
                pending.remove(t)
        return result

    def _filter_ready(self, tasks: list[TaskModel]) -> tuple[list[TaskModel], list[TaskModel]]:
        """Split tasks into ready (all deps completed) and skipped (deps not yet complete)."""
        completed_ids = {t.id for t in self._tasks if t.status == "completed"}
        ready: list[TaskModel] = []
        skipped: list[TaskModel] = []
        for t in tasks:
            deps = [d for d in t.dependencies if d in {x.id for x in self._tasks}]
            if all(d in completed_ids for d in deps):
                ready.append(t)
            else:
                skipped.append(t)
        return ready, skipped

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self._failed_task_descriptions)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("planning_complete", rationale=plan.rationale, task_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            log.warning("no_tasks_planned")
            return

        if self.cycles == 1:
            self._tasks = list(plan.tasks)
        else:
            self._tasks.extend(plan.tasks)

        # Detect and break dependency cycles
        self._detect_cycles(self._tasks)

        # Sort tasks topologically (dependencies first)
        ordered = self._topological_sort(self._tasks)

        # Only execute tasks that are in this plan; filter by dependency readiness
        ready, skipped = self._filter_ready(ordered)
        for t in skipped:
            unmet = [d for d in t.dependencies if d not in {x.id for x in self._tasks if x.status == "completed"}]
            console.print(f"[yellow]Task {t.id} skipped: waiting on dependencies {unmet}[/yellow]")
            log.warning("task_skipped", task_id=t.id, unmet_dependencies=unmet)

        max_parallel = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(max_parallel)

        async def run_with_semaphore(task: TaskModel) -> Any:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [run_with_semaphore(task) for task in ready]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(ready, results):
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

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", summary=test_result.summary)
            self._failed_task_descriptions = []
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._failed_task_descriptions = test_result.failures

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self._tasks
