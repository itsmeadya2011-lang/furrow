from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.goal = goal
        self.focus = goal
        self.client = client or LLMClient()
        self.settings = settings or default_settings
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.last_test_result: TestResult | None = None

    def _emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001, S110 - intentional: broken listener must not crash loop
                pass

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while True:
            self.cycles += 1
            self._emit({"type": "cycle", "cycle": self.cycles})
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            plan = await self.planner.plan(self.focus)
            self._emit({"type": "plan", "cycle": self.cycles, "plan": plan.model_dump()})
            console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

            if not plan.tasks:
                if self.last_test_result is not None and self.last_test_result.passed:
                    self._emit({"type": "done", "reason": "plan empty, tests passing", "cycles": self.cycles})
                    console.print("[bold green]Plan empty and tests passing. Halting.[/bold green]")
                    break
                else:
                    self._emit({"type": "done", "reason": "plan empty", "cycles": self.cycles})
                    console.print("[yellow]Plan empty. Halting.[/yellow]")
                    break

            for task in plan.tasks:
                self.tasks.append(task)

            max_parallel = self.settings.max_parallel_tasks
            if max_parallel <= 0:
                semaphore = asyncio.Semaphore(len(plan.tasks))
            else:
                semaphore = asyncio.Semaphore(max_parallel)

            completed_in_cycle = set()
            failed_in_cycle = set()
            skipped_in_cycle = set()

            cycle_task_ids = {task.id for task in plan.tasks}

            while True:
                ready = []
                for task in plan.tasks:
                    if task.id in completed_in_cycle or task.id in failed_in_cycle or task.id in skipped_in_cycle:
                        continue
                    deps_satisfied = True
                    dep_failed = False
                    for dep_id in task.dependencies:
                        if dep_id in failed_in_cycle:
                            dep_failed = True
                            break
                        if dep_id not in completed_in_cycle and dep_id in cycle_task_ids:
                            deps_satisfied = False
                            break
                    if dep_failed:
                        skipped_in_cycle.add(task.id)
                        task.status = "skipped"
                        self._emit({
                            "type": "task_start",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "description": task.description,
                        })
                        self._emit({
                            "type": "task_done",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "status": "skipped",
                            "result": "dependency failed",
                        })
                        console.print(f"[yellow]Task {task.id} skipped: dependency failed[/yellow]")
                    elif deps_satisfied:
                        ready.append(task)

                if not ready:
                    break

                async def run_task(task: TaskModel, semaphore=semaphore) -> tuple[TaskModel, str | Exception]:
                    async with semaphore:
                        self._emit({
                            "type": "task_start",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "description": task.description,
                        })
                        console.print(f"[bold yellow]Running task {task.id}: {task.description}[/bold yellow]")
                        worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                        try:
                            result = await worker.run()
                            return task, result
                        except Exception as e:  # noqa: BLE001 - intentional: catch worker errors as results
                            return task, e

                results = await asyncio.gather(*[run_task(t) for t in ready], return_exceptions=False)

                for task, result in results:
                    if isinstance(result, Exception):
                        task.status = "failed"
                        task.result = str(result)
                        failed_in_cycle.add(task.id)
                        self._emit({
                            "type": "task_done",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "status": "failed",
                            "result": str(result),
                        })
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = result
                        completed_in_cycle.add(task.id)
                        self._emit({
                            "type": "task_done",
                            "cycle": self.cycles,
                            "task_id": task.id,
                            "status": "completed",
                            "result": result,
                        })
                        console.print(f"[green]Task {task.id} completed[/green]")

            test_result = await self.tester.run(self.goal, plan.tasks)
            self.last_test_result = test_result
            self._emit({"type": "tests", "cycle": self.cycles, "result": test_result.model_dump()})

            if test_result.passed:
                console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            else:
                console.print(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    console.print(f"  • {failure}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

            if test_result.passed:
                self._emit({"type": "done", "reason": "tests passing", "cycles": self.cycles})
                console.print("[bold green]Tests passing. Goal complete.[/bold green]")
                break

            self.focus = (
                "Fix failing tests for the original goal.\n"
                f"Original goal: {self.goal}\n"
                "Failures:\n" + "\n".join(test_result.failures)
            )

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit({"type": "done", "reason": f"max_cycles ({self.settings.max_cycles}) reached", "cycles": self.cycles})
                console.print(f"[bold red]Max cycles ({self.settings.max_cycles}) reached. Halting.[/bold red]")
                break

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _is_done(self) -> bool:
        return bool(self.last_test_result and self.last_test_result.passed)