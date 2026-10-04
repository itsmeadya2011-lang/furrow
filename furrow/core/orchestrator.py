from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
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
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger()

BLOCKING_STATUSES = ("failed", "blocked")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[dict], Any] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.on_event = on_event
        self.last_test_result: TestResult | None = None
        self.fix_requests: list[str] = []
        self._complete = False
        self._load_state()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return Path(self.client.settings.state_file)

    def _load_state(self) -> None:
        path = self._state_path()
        try:
            if not path.exists():
                return
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise TypeError("state file must contain a JSON object")

            goal = data.get("goal", self.goal)
            original_goal = data.get("original_goal", self.original_goal)
            cycles = int(data.get("cycles", 0))
            tasks = [TaskModel(**task) for task in data.get("tasks", [])]
            raw_test_result = data.get("last_test_result")
            last_test_result = TestResult(**raw_test_result) if raw_test_result else None
            fix_requests = [str(item) for item in data.get("fix_requests", [])]
            complete = bool(data.get("complete", False))
        except Exception as exc:  # corrupt or unreadable state -> start fresh
            log.warning("orchestrator.state_load_failed", path=str(path), error=str(exc))
            return

        self.goal = goal
        self.original_goal = original_goal
        self.cycles = cycles
        self.tasks = tasks
        self.last_test_result = last_test_result
        self.fix_requests = fix_requests
        self._complete = complete
        log.info(
            "orchestrator.state_loaded",
            path=str(path),
            cycles=self.cycles,
            tasks=len(self.tasks),
            fix_requests=len(self.fix_requests),
        )

    def _save_state(self) -> None:
        path = self._state_path()
        state = {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "cycles": self.cycles,
            "tasks": [task.model_dump() for task in self.tasks],
            "last_test_result": self.last_test_result.model_dump() if self.last_test_result else None,
            "fix_requests": self.fix_requests,
            "complete": self._complete,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            log.warning("orchestrator.state_save_failed", path=str(path), error=str(exc))
            return
        log.debug(
            "orchestrator.state_saved",
            path=str(path),
            cycles=self.cycles,
            tasks=len(self.tasks),
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        try:
            result = self.on_event(event)
            if inspect.isawaitable(result):
                try:
                    asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    log.debug("orchestrator.event_awaitable_dropped", event_type=event.get("type"))
        except Exception as exc:
            log.warning("orchestrator.event_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        try:
            while True:
                self.cycles += 1
                log.info("orchestrator.cycle_start", cycle=self.cycles, goal=self.goal)
                self._emit({"type": "cycle_start", "cycle": self.cycles, "goal": self.goal})
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

                max_cycles = self.client.settings.max_cycles
                if max_cycles > 0 and self.cycles > max_cycles:
                    console.print(f"[yellow]Reached max cycles ({max_cycles}). Halting.[/yellow]")
                    self._emit(
                        {"type": "halted", "reason": "max_cycles", "cycles": self.cycles}
                    )
                    break

                await self._cycle()
                self._save_state()
                log.info("orchestrator.cycle_complete", cycle=self.cycles)
                self._emit({"type": "cycle_complete", "cycle": self.cycles})

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    self._emit({"type": "halted", "reason": "complete", "cycles": self.cycles})
                    break
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            console.print("[yellow]Stopped by user.[/yellow]")
            self._emit({"type": "halted", "reason": "interrupted", "cycles": self.cycles})
            self._save_state()
            log.info("orchestrator.interrupted", cycles=self.cycles)
            return

    async def _cycle(self) -> None:
        planning_goal = self.original_goal
        if self.fix_requests:
            planning_goal += "\n\nOutstanding failures to fix:\n" + "\n".join(self.fix_requests)

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(planning_goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            {
                "type": "plan",
                "cycle": self.cycles,
                "rationale": plan.rationale,
                "task_ids": [task.id for task in plan.tasks],
            }
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._complete = True
            return

        self._merge_tasks(plan.tasks)
        await self._execute(plan.tasks)
        self._sync_task_state(plan.tasks)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.original_goal, plan.tasks)
        self.last_test_result = test_result
        self._emit(
            {
                "type": "test_result",
                "cycle": self.cycles,
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
            }
        )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.fix_requests.extend(test_result.failures)
            log.warning(
                "orchestrator.tests_failed",
                cycle=self.cycles,
                summary=test_result.summary,
                failures=len(test_result.failures),
            )

    # ------------------------------------------------------------------
    # Task bookkeeping
    # ------------------------------------------------------------------

    def _merge_tasks(self, plan_tasks: list[TaskModel]) -> None:
        known = {task.id: task for task in self.tasks}
        for task in plan_tasks:
            current = known.get(task.id)
            if current is None:
                self.tasks.append(task)
                known[task.id] = task
                continue
            current.description = task.description
            current.files = list(task.files)
            current.dependencies = list(task.dependencies)
            current.status = "pending"
            current.result = None

    def _sync_task_state(self, plan_tasks: list[TaskModel]) -> None:
        """Copy execution status/results from freshly planned tasks onto tracked tasks."""
        tracked = {task.id: task for task in self.tasks}
        for task in plan_tasks:
            current = tracked.get(task.id)
            if current is None or current is task:
                continue
            current.status = task.status
            current.result = task.result

    # ------------------------------------------------------------------
    # Dependency-aware parallel execution
    # ------------------------------------------------------------------

    async def _execute(self, plan_tasks: list[TaskModel]) -> None:
        remaining: dict[str, TaskModel] = {}
        by_id: dict[str, TaskModel] = {}
        for task in plan_tasks:
            by_id[task.id] = task
            remaining.setdefault(task.id, task)

        if not remaining:
            return

        semaphore = asyncio.Semaphore(max(1, self.client.settings.max_parallel_tasks))

        async def run_task(task: TaskModel) -> str:
            task.status = "running"
            self._emit(
                {
                    "type": "task_start",
                    "cycle": self.cycles,
                    "task_id": task.id,
                    "description": task.description,
                }
            )
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        def blocking_dependencies(task: TaskModel) -> list[str]:
            return [
                dep
                for dep in task.dependencies
                if dep in by_id and by_id[dep].status in BLOCKING_STATUSES
            ]

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            while remaining:
                for task in [t for t in remaining.values() if blocking_dependencies(t)]:
                    remaining.pop(task.id, None)
                    blockers = blocking_dependencies(task)
                    task.status = "blocked"
                    task.result = f"Blocked by failed tasks: {', '.join(blockers)}"
                    console.print(f"[yellow]Task {task.id} blocked by {', '.join(blockers)}[/yellow]")
                    self._emit(
                        {
                            "type": "task_blocked",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "blocked_by": blockers,
                        }
                    )

                wave = [
                    task
                    for task in remaining.values()
                    if all(
                        dep not in by_id or by_id[dep].status == "completed"
                        for dep in task.dependencies
                    )
                ]

                if not wave:
                    if remaining:
                        console.print("[red]Unresolvable dependencies; aborting remaining tasks.[/red]")
                        for task in remaining.values():
                            task.status = "failed"
                            task.result = "Unresolvable dependencies"
                            log.warning(
                                "orchestrator.task_blocked_forever",
                                cycle=self.cycles,
                                task_id=task.id,
                            )
                            self._emit(
                                {
                                    "type": "task_failed",
                                    "cycle": self.cycles,
                                    "task_id": task.id,
                                    "error": task.result,
                                }
                            )
                        remaining.clear()
                    break

                results = await asyncio.gather(
                    *(run_task(task) for task in wave), return_exceptions=True
                )

                for task, result in zip(wave, results):
                    remaining.pop(task.id, None)
                    if isinstance(result, BaseException):
                        task.status = "failed"
                        task.result = str(result)
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                        log.warning(
                            "orchestrator.task_failed",
                            cycle=self.cycles,
                            task_id=task.id,
                            error=str(result),
                        )
                        self._emit(
                            {
                                "type": "task_failed",
                                "cycle": self.cycles,
                                "task_id": task.id,
                                "error": str(result),
                            }
                        )
                    else:
                        text = result if isinstance(result, str) else str(result)
                        task.status = "completed"
                        task.result = text
                        console.print(f"[green]Task {task.id} completed[/green]")
                        log.info(
                            "orchestrator.task_completed",
                            cycle=self.cycles,
                            task_id=task.id,
                            result_length=len(text),
                        )
                        self._emit(
                            {
                                "type": "task_complete",
                                "cycle": self.cycles,
                                "task_id": task.id,
                                "result_length": len(text),
                            }
                        )

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def _is_done(self) -> bool:
        if self._complete:
            return True
        if not self.tasks:
            return False
        if any(
            task.status in ("pending", "running", "blocked", "failed") for task in self.tasks
        ):
            return False
        return self.last_test_result is not None and self.last_test_result.passed
