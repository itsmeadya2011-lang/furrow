from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.panel import Panel

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult, get_settings
from furrow.events import EventBus, EventType
from furrow.llm import LLMClient

console = Console()

FEEDBACK_LIMIT = 5
REASON_STOPPED = "stopped"
REASON_MAX_CYCLES = "max_cycles"
REASON_STALLED = "stalled"
REASON_GOAL_COMPLETE = "goal_complete"


@dataclass(slots=True)
class RunSummary:
    """Outcome of a completed run, returned by ``Orchestrator.run``."""

    cycles: int
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    tests_passed: bool
    halted_reason: str


class Orchestrator:
    """Runs the plan -> execute -> test loop until a halt condition is met."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.goal = goal
        self.focus = goal
        self.settings = settings or get_settings()
        self.client = client or LLMClient(self.settings)
        self.bus = bus if bus is not None else EventBus(self.settings.max_event_history)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: dict[str, TaskModel] = {}
        self.test_results: list[TestResult] = []
        self._stop = asyncio.Event()
        self._halted_reason: str | None = None
        self._running = False
        self._stalled = 0
        self._progress = False
        self._feedback: list[str] = []
        self._last_test_passed: bool | None = None

    async def run(self) -> RunSummary:
        self._running = True
        self.bus.emit(EventType.RUN_START, f"Goal: {self.goal}", cycle=self.cycles)
        banner = f"[bold green]Furrow[/bold green]\nGoal: {self.goal}"
        console.print(Panel.fit(banner, title="Furrow"))
        try:
            while True:
                if self._stop.is_set():
                    self._halted_reason = REASON_STOPPED
                    break
                self.cycles += 1
                self.bus.emit(
                    EventType.CYCLE_START, f"Cycle {self.cycles}", cycle=self.cycles
                )
                console.print(f"\n[bold cyan]=== Cycle {self.cycles} ===[/bold cyan]")
                plan = await self._guarded_cycle()
                self._record_progress()
                self.bus.emit(
                    EventType.CYCLE_END,
                    f"Cycle {self.cycles} finished",
                    cycle=self.cycles,
                    stalled_cycles=self._stalled,
                )
                self._persist()
                if self._stop.is_set():
                    self._halted_reason = REASON_STOPPED
                    break
                if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                    self._halted_reason = REASON_MAX_CYCLES
                    break
                stalled_limit = self.settings.max_stalled_cycles
                if stalled_limit > 0 and self._stalled >= stalled_limit:
                    self._halted_reason = REASON_STALLED
                    break
                if plan is not None and not plan.tasks:
                    self._halted_reason = REASON_GOAL_COMPLETE
                    break
        finally:
            self._running = False
            self.bus.emit(
                EventType.RUN_END,
                f"Furrow halted: {self._halted_reason}",
                cycle=self.cycles,
                halted_reason=self._halted_reason,
            )
            console.print(f"[bold]Furrow halted: {self._halted_reason}[/bold]")
            self._persist()
        return self._summary()

    async def stop(self) -> None:
        """Request a graceful stop; ``run`` exits at the next safe point."""
        self._stop.set()

    @property
    def halted_reason(self) -> str | None:
        return self._halted_reason

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the current run state."""
        return {
            "goal": self.goal,
            "focus": self.focus,
            "cycles": self.cycles,
            "halted_reason": self._halted_reason,
            "running": self._running,
            "tasks": [
                {
                    "id": task.id,
                    "description": task.description,
                    "status": task.status.value,
                    "attempts": task.attempts,
                    "files": list(task.files),
                    "dependencies": list(task.dependencies),
                    "result": task.result,
                }
                for task in self.tasks.values()
            ],
            "test_results": [
                {
                    "passed": result.passed,
                    "summary": result.summary,
                    "failures": list(result.failures),
                    "commands": list(result.commands),
                }
                for result in self.test_results
            ],
            "usage": getattr(self.client, "usage", {}),
        }

    async def _guarded_cycle(self) -> Plan | None:
        try:
            return await self._cycle()
        except Exception as exc:
            self._record_error("Cycle failed", exc)
            return None

    async def _cycle(self) -> Plan | None:
        self._progress = False
        try:
            plan = await self.planner.plan(self.focus)
        except Exception as exc:
            self._record_error("Planner failed", exc)
            return None
        self.bus.emit(
            EventType.PLAN,
            f"Planned {len(plan.tasks)} task(s)",
            cycle=self.cycles,
            rationale=plan.rationale,
            tasks=[
                {
                    "id": task.id,
                    "description": task.description,
                    "files": task.files,
                    "dependencies": task.dependencies,
                }
                for task in plan.tasks
            ],
        )
        console.print(f"[bold]Plan ({len(plan.tasks)} tasks):[/bold] {plan.rationale}")
        if not plan.tasks:
            self.bus.emit(
                EventType.LOG, "Empty plan received; goal appears complete", cycle=self.cycles
            )
            return plan
        tasks = self._merge_plan(plan)
        await self._execute(tasks)
        tester = TesterAgent(client=self.client, settings=self.settings)
        try:
            test_result = await tester.run(self.goal, tasks)
        except Exception as exc:
            self._record_error("Tester failed", exc)
            return plan
        self.test_results.append(test_result)
        flipped = self._last_test_passed is None or test_result.passed != self._last_test_passed
        self._last_test_passed = test_result.passed
        self.bus.emit(
            EventType.TEST_RESULT,
            test_result.summary,
            cycle=self.cycles,
            passed=test_result.passed,
            summary=test_result.summary,
            failures=test_result.failures,
            commands=test_result.commands,
        )
        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self._feedback.clear()
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  - {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self._feedback.extend(test_result.failures)
            self._feedback = self._feedback[-FEEDBACK_LIMIT:]
        self._update_focus()
        if flipped:
            self._progress = True
        return plan

    def _merge_plan(self, plan: Plan) -> list[TaskModel]:
        merged: list[TaskModel] = []
        seen: set[str] = set()
        fresh = 0
        for incoming in plan.tasks:
            if incoming.id in seen:
                continue
            seen.add(incoming.id)
            existing = self.tasks.get(incoming.id)
            if existing is None:
                self.tasks[incoming.id] = incoming
                fresh += 1
            else:
                existing.description = incoming.description
                existing.files = incoming.files
                existing.dependencies = incoming.dependencies
            merged.append(self.tasks[incoming.id])
        if fresh:
            self._progress = True
            self.bus.emit(
                EventType.LOG,
                f"{fresh} new task(s) planned; {len(self.tasks)} tracked overall",
                cycle=self.cycles,
                new_tasks=fresh,
                tracked_tasks=len(self.tasks),
            )
        return merged

    async def _execute(self, tasks: list[TaskModel]) -> None:
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        for wave in self._plan_waves(tasks):
            coroutines = [self._run_task(task, semaphore) for task in wave]
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            for task, result in zip(wave, results, strict=True):
                if isinstance(result, BaseException):
                    task.status = TaskStatus.FAILED
                    task.result = str(result)
                    self._emit_task_end(task, TaskStatus.FAILED.value, error=str(result))
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.result = result
                    # FIX (test suite): `WorkerAgent.run` reports a failure by RETURNING a
                    # description (it never raises for an expected fault), so a non-exception
                    # result is not proof of success. Trust the status the worker already
                    # set: otherwise every broken worker counted as progress, `summary()` said
                    # "completed", and a permanently failing run never stalled.
                    if task.status is not TaskStatus.COMPLETED:
                        self._emit_task_end(
                            task, TaskStatus.FAILED.value, error=str(result)
                        )
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = TaskStatus.COMPLETED
                        self._progress = True
                        self._emit_task_end(task, TaskStatus.COMPLETED.value)
                        console.print(f"[green]Task {task.id} completed[/green]")

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            task.status = TaskStatus.IN_PROGRESS
            task.attempts += 1
            self.bus.emit(
                EventType.TASK_START,
                f"Task {task.id}: {task.description}",
                cycle=self.cycles,
                id=task.id,
                description=task.description,
                status=TaskStatus.IN_PROGRESS.value,
            )
            console.print(f"[cyan]Task {task.id} started[/cyan]")
            worker = WorkerAgent(
                task=task, client=self.client, settings=self.settings, bus=self.bus
            )
            return await worker.run()

    def _plan_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        by_id = {task.id: task for task in tasks}
        unknown = sorted({dep for task in tasks for dep in task.dependencies if dep not in by_id})
        if unknown:
            self.bus.emit(
                EventType.LOG,
                f"Unknown dependencies treated as satisfied: {', '.join(unknown)}",
                cycle=self.cycles,
                unknown_dependencies=unknown,
            )
        pending = {task.id: {dep for dep in task.dependencies if dep in by_id} for task in tasks}
        waves: list[list[TaskModel]] = []
        while pending:
            ready = sorted(task_id for task_id, deps in pending.items() if not deps)
            if not ready:
                self.bus.emit(
                    EventType.LOG,
                    "Dependency cycle detected; running remaining tasks together",
                    cycle=self.cycles,
                    task_ids=sorted(pending),
                )
                waves.append([by_id[task_id] for task_id in sorted(pending)])
                break
            for task_id in ready:
                del pending[task_id]
            for deps in pending.values():
                deps.difference_update(ready)
            waves.append([by_id[task_id] for task_id in ready])
        return waves

    def _emit_task_end(self, task: TaskModel, status: str, error: str | None = None) -> None:
        data: dict[str, Any] = {
            "id": task.id,
            "description": task.description,
            "status": status,
        }
        if error is not None:
            data["error"] = error
        self.bus.emit(EventType.TASK_END, f"Task {task.id} {status}", cycle=self.cycles, **data)

    def _record_progress(self) -> None:
        if self._progress:
            self._stalled = 0
        else:
            self._stalled += 1
        self._progress = False

    def _update_focus(self) -> None:
        if self._feedback:
            feedback = "\n".join(f"- {failure}" for failure in self._feedback)
            self.focus = f"{self.goal}\n\nCurrent failing tests:\n{feedback}"
        else:
            self.focus = self.goal

    def _record_error(self, context: str, exc: BaseException) -> None:
        self.bus.emit(
            EventType.ERROR,
            f"{context}: {exc}",
            cycle=self.cycles,
            error=str(exc),
            context=context,
        )
        console.print(f"[red]{context}: {exc}[/red]")

    def _persist(self) -> None:
        state_file = self.settings.state_file
        if state_file is None:
            return
        try:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(self.snapshot(), indent=2)
            temp_file = state_file.with_name(state_file.name + ".tmp")
            with open(temp_file, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temp_file, state_file)
        except Exception as exc:
            self._record_error("State persistence failed", exc)

    def _summary(self) -> RunSummary:
        tasks = list(self.tasks.values())
        return RunSummary(
            cycles=self.cycles,
            tasks_total=len(tasks),
            tasks_completed=sum(1 for task in tasks if task.status == TaskStatus.COMPLETED),
            tasks_failed=sum(1 for task in tasks if task.status == TaskStatus.FAILED),
            tests_passed=bool(self.test_results) and self.test_results[-1].passed,
            halted_reason=self._halted_reason or "unknown",
        )
