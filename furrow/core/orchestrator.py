from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult, settings as default_settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or getattr(self.client, "settings", default_settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.plan: Plan | None = None
        self.tasks: list[TaskModel] = []
        self.latest_test_result: TestResult | None = None
        self._halt_requested = False

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}", title="Furrow"))
        while not self._cycle_limit_reached():
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self._halt_requested:
                console.print("[yellow]Furrow halted because no safe progress is possible.[/yellow]")
                break
            if self._cycle_limit_reached():
                console.print(f"[yellow]Cycle limit of {self.settings.max_cycles} reached. Halting.[/yellow]")
                break

    async def _cycle(self) -> None:
        planning_goal = self._get_planning_goal()
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(planning_goal)
        except Exception as exc:
            self._record_planner_failure(exc)
            return

        self._merge_plan(plan)
        console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))

        pending_tasks = [task for task in self.tasks if task.status == TaskStatus.PENDING]
        if not pending_tasks and not self._tasks_are_complete():
            self._halt_requested = True
            return

        if pending_tasks:
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
                results = await asyncio.gather(
                    *(self._run_task(task, semaphore) for task in pending_tasks),
                    return_exceptions=True,
                )
                for task, result in zip(pending_tasks, results):
                    if isinstance(result, Exception):
                        self._fail_task(task, result)

        try:
            with Status("[bold yellow]Testing...", console=console):
                test_result = await TesterAgent(client=self.client).run(
                    self.original_goal, self.tasks
                )
        except Exception as exc:
            test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
            self._halt_requested = True
        self.latest_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            if not pending_tasks and not self._tasks_are_complete():
                self._halt_requested = True
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            if not pending_tasks:
                self._halt_requested = True
            else:
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            task.status = TaskStatus.RUNNING
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
            except Exception as exc:
                self._fail_task(task, exc)
            else:
                task.status = TaskStatus.COMPLETED
                task.result = str(result)
                console.print(f"[green]Task {task.id} completed[/green]")

    def _fail_task(self, task: TaskModel, exc: Exception) -> None:
        task.status = TaskStatus.FAILED
        task.result = str(exc)
        console.print(f"[red]Task {task.id} failed: {exc}[/red]")

    def _merge_plan(self, plan: Plan) -> None:
        merged_tasks = list(self.tasks)
        for incoming_task in plan.tasks:
            existing_task = next(
                (task for task in merged_tasks if task.id == incoming_task.id),
                None,
            )
            if existing_task is None:
                incoming_task.status = TaskStatus.PENDING
                incoming_task.result = None
                merged_tasks.append(incoming_task)
                continue

            existing_task.description = incoming_task.description
            existing_task.files = list(incoming_task.files)
            existing_task.dependencies = list(incoming_task.dependencies)
            if existing_task.status == TaskStatus.FAILED:
                existing_task.status = TaskStatus.PENDING
                existing_task.result = None
            elif existing_task.status == TaskStatus.COMPLETED and self.latest_test_result is not None and not self.latest_test_result.passed:
                existing_task.status = TaskStatus.PENDING
                existing_task.result = None

        plan.tasks = merged_tasks
        self.tasks = merged_tasks
        self.plan = plan

    def _record_planner_failure(self, exc: Exception) -> None:
        task = TaskModel(
            id=f"planner-cycle-{self.cycles + 1}",
            description="Create a plan for the goal",
            status=TaskStatus.FAILED,
            result=str(exc),
        )
        self.tasks.append(task)
        self.plan = Plan(tasks=list(self.tasks), rationale=str(exc))
        self.latest_test_result = TestResult(passed=False, summary=str(exc), failures=[str(exc)])
        self._halt_requested = True
        console.print(f"[red]Planning failed: {exc}[/red]")

    def _get_planning_goal(self) -> str:
        if self.latest_test_result is None or self.latest_test_result.passed:
            return self.original_goal

        failures = "\n".join(self.latest_test_result.failures) or self.latest_test_result.summary
        completed_work = "\n".join(
            f"- {task.id}: {task.result or ''}"
            for task in self.tasks
            if task.status == TaskStatus.COMPLETED and (task.result or "").strip()
        )
        if completed_work:
            return (
                f"{self.original_goal}\n\n"
                f"Test failures to fix:\n{failures}\n\n"
                f"Completed work:\n{completed_work}"
            )
        return f"{self.original_goal}\n\nTest failures to fix:\n{failures}"

    def _tasks_are_complete(self) -> bool:
        return bool(self.tasks) and all(task.status == TaskStatus.COMPLETED for task in self.tasks)

    def _cycle_limit_reached(self) -> bool:
        return self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles

    def _is_done(self) -> bool:
        return (
            not self._halt_requested
            and self.latest_test_result is not None
            and self.latest_test_result.passed
            and self._tasks_are_complete()
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
