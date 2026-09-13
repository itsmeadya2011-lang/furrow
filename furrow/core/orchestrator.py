from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], Awaitable[None]] | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        self._original_goal = goal
        self._test_context: str = ""
        if client is not None:
            self.client = client
            self.settings = client.settings
        else:
            self.client = LLMClient(settings=settings_obj)
            self.settings = settings_obj or self.client.settings
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self._tasks: dict[str, TaskModel] = {}
        self._cycles = 0
        self._last_pending_plan_hash: str = ""
        self._no_progress_count = 0
        self._last_test_passed: bool = False
        self._worker_failures: dict[str, str] = {}

    @property
    def goal(self) -> str:
        return self._original_goal

    @property
    def cycles(self) -> int:
        return self._cycles

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self._original_goal}", title="Furrow"))
        await self._emit(f"Furrow started. Goal: {self._original_goal}")

        max_cycles = self.settings.max_cycles
        unlimited = max_cycles == 0

        try:
            while unlimited or self._cycles < max_cycles:
                self._cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self._cycles} ═══[/bold cyan]")
                await self._emit(f"Cycle {self._cycles}")

                progress_made = await self._cycle()

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    await self._emit("Goal complete. Halting.")
                    break

                if not progress_made:
                    self._no_progress_count += 1
                    if self._no_progress_count >= 2:
                        console.print("[yellow]No progress for 2 cycles. Stopping.[/yellow]")
                        await self._emit("No progress for 2 cycles. Stopping.")
                        break
                else:
                    self._no_progress_count = 0

            if not unlimited and self._cycles >= max_cycles and not self._is_done():
                console.print(f"[yellow]Reached max_cycles ({max_cycles}). Halting.[/yellow]")
                await self._emit(f"Reached max_cycles ({max_cycles}). Halting.")

        except asyncio.CancelledError:
            console.print("[yellow]Orchestrator cancelled.[/yellow]")
            raise

    async def _emit(self, message: str) -> None:
        if self.on_event:
            await self.on_event(message)

    def _build_planner_goal(self) -> str:
        if not self._test_context and not self._worker_failures:
            return self._original_goal
        parts = [self._original_goal]
        if self._test_context:
            parts.append(f"=== Test Failure Context ===\n{self._test_context}\n=== End Context ===")
        if self._worker_failures:
            worker_lines = "\n".join(f"{tid}: {err}" for tid, err in sorted(self._worker_failures.items()))
            parts.append(f"=== Worker Failure Context ===\n{worker_lines}\n=== End Context ===")
        return "\n\n".join(parts)

    def _pending_plan_hash(self, plan: Plan) -> str:
        pending_tasks = [
            {"id": t.id, "description": t.description, "dependencies": t.dependencies}
            for t in plan.tasks
            if t.id not in self._tasks or self._tasks[t.id].status != "completed"
        ]
        failure_context = self._test_context
        worker_failures = sorted(f"{tid}:{err}" for tid, err in self._worker_failures.items())
        return json.dumps(
            {"pending": pending_tasks, "test_context": failure_context, "worker_failures": worker_failures},
            sort_keys=True,
        )

    async def _cycle(self) -> bool:
        working_goal = self._build_planner_goal()

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(working_goal)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        await self._emit(f"Plan: {len(plan.tasks)} tasks")

        self._merge_plan(plan)

        current_pending_hash = self._pending_plan_hash(plan)
        if current_pending_hash == self._last_pending_plan_hash:
            # Repeated plan means no new work to execute; skip worker execution
            # but still run tests to capture state changes and update context.
            console.print("[yellow]Repeated pending plan detected. Skipping task execution.[/yellow]")
            await self._emit("Repeated pending plan detected. Skipping task execution.")
            progress_made = False
        else:
            self._last_pending_plan_hash = current_pending_hash
            progress_made = await self._execute_pending_tasks()

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self._original_goal, list(self._tasks.values()))

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            await self._emit(f"Tests passed: {test_result.summary}")
            self._test_context = ""
            self._last_test_passed = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
                await self._emit(f"Test failure: {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            await self._emit("Tests failed. Will attempt fix in next cycle.")
            self._test_context = "\n".join(test_result.failures)
            self._last_test_passed = False

        return progress_made

    def _merge_plan(self, plan: Plan) -> None:
        has_test_context = bool(self._test_context)
        seen_ids: set[str] = set()
        for task in plan.tasks:
            # Deterministically handle duplicate task IDs within one plan:
            # keep the first occurrence and skip subsequent duplicates so we
            # never unexpectedly overwrite an already-stored task.
            if task.id in seen_ids:
                console.print(f"[yellow]Duplicate task id '{task.id}' in plan; keeping first occurrence.[/yellow]")
                continue
            seen_ids.add(task.id)

            if task.id in self._tasks:
                existing = self._tasks[task.id]
                existing.description = task.description
                existing.files = task.files
                existing.dependencies = task.dependencies
                # Preserve existing retry semantics: only reset to pending when
                # there is a test/worker failure context that warrants re-execution.
                if has_test_context or task.id in self._worker_failures:
                    existing.status = "pending"
                    existing.result = None
            else:
                # Newly planned tasks must never inherit a planner-provided
                # status/result that could mark them complete without execution.
                task.status = "pending"
                task.result = None
                self._tasks[task.id] = task

    async def _execute_pending_tasks(self) -> bool:
        pending_tasks = [t for t in self._tasks.values() if t.status == "pending"]
        if not pending_tasks:
            return False

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        progress_made = False

        async def run_task(task: TaskModel) -> None:
            nonlocal progress_made
            async with semaphore:
                try:
                    worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                    result = await worker.run()
                    task.status = "completed"
                    task.result = result
                    progress_made = True
                    self._remove_worker_failure_from_context(task.id)
                    console.print(f"[green]Task {task.id} completed[/green]")
                    await self._emit(f"Task {task.id} completed")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    task.status = "failed"
                    task.result = str(e)
                    self._add_worker_failure_to_context(task.id, str(e))
                    progress_made = True
                    console.print(f"[red]Task {task.id} failed: {e}[/red]")
                    await self._emit(f"Task {task.id} failed: {e}")

        while True:
            ready_tasks = []
            for task in pending_tasks:
                if self._is_ready(task):
                    ready_tasks.append(task)

            if not ready_tasks:
                for task in pending_tasks:
                    if task.status == "pending":
                        dep_status = self._get_dependency_status(task)
                        if dep_status == "failed":
                            result = f"Dependency failed: {self._get_failed_dependency(task)}"
                            task.status = "failed"
                            task.result = result
                            self._add_worker_failure_to_context(task.id, result)
                            console.print(f"[red]Task {task.id} blocked: {result}[/red]")
                            await self._emit(f"Task {task.id} blocked: {result}")
                            progress_made = True
                        elif dep_status == "unknown" or dep_status == "cyclic":
                            result = f"Dependency issue: {dep_status}"
                            task.status = "failed"
                            task.result = result
                            self._add_worker_failure_to_context(task.id, result)
                            console.print(f"[red]Task {task.id} blocked: {result}[/red]")
                            await self._emit(f"Task {task.id} blocked: {result}")
                            progress_made = True
                break

            tasks_to_run = [run_task(task) for task in ready_tasks]
            await asyncio.gather(*tasks_to_run, return_exceptions=True)

            pending_tasks = [t for t in pending_tasks if t.status == "pending"]
            if not pending_tasks:
                break

        return progress_made

    def _add_worker_failure_to_context(self, task_id: str, error: str) -> None:
        self._worker_failures[task_id] = error

    def _remove_worker_failure_from_context(self, task_id: str) -> None:
        self._worker_failures.pop(task_id, None)

    def _is_ready(self, task: TaskModel) -> bool:
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if dep_task is None:
                return False
            if dep_task.status != "completed":
                return False
        return True

    def _get_dependency_status(self, task: TaskModel) -> str:
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if dep_task is None:
                return "unknown"
            if dep_task.status == "failed":
                return "failed"
            if dep_task.status == "pending":
                return "cyclic"
        return "completed"

    def _get_failed_dependency(self, task: TaskModel) -> str:
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if dep_task and dep_task.status == "failed":
                return dep_id
        return "unknown"

    def _is_done(self) -> bool:
        if not self._last_test_passed:
            return False
        if any(t.status in ("failed", "pending") for t in self._tasks.values()):
            return False
        return True

    def _get_tasks(self) -> list[TaskModel]:
        return list(self._tasks.values())
