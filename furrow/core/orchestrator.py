from __future__ import annotations

import asyncio

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger("furrow.orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.current_tasks: list[TaskModel] = []
        self.last_test_passed: bool = False

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_start", goal=self.goal)
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            plan = await self._plan()
            if plan is None:
                if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                    console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                    logger.warning("max_cycles_reached", cycles=self.cycles)
                    break
                continue

            await self._cycle(plan)

            if self._is_done():
                console.print("[bold green]Goal complete. Halted.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
                logger.warning("max_cycles_reached", cycles=self.cycles)
                break

    async def _plan(self) -> Plan | None:
        try:
            with Status("[bold yellow]Planning...", console=console):
                plan = await self.planner.plan(self.goal)
        except Exception as e:
            console.print(f"[red]Planning failed: {e}[/red]")
            logger.error("planning_failed", error=str(e))
            return None
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("planning_complete", task_count=len(plan.tasks))
        return plan

    async def _cycle(self, plan: Plan) -> None:
        self.current_tasks = plan.tasks
        self.last_test_passed = False

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned")
            return

        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        waves = self._topological_sort(plan.tasks)
        logger.info(
            "executing_tasks",
            total_tasks=len(plan.tasks),
            waves=len(waves),
            max_parallel=self.settings.max_parallel_tasks,
        )

        async def _execute(task: TaskModel) -> str:
            async with semaphore:
                logger.info("task_start", task_id=task.id, description=task.description)
                console.print(f"[bold]Running task {task.id}: {task.description}[/bold]")
                return await WorkerAgent(
                    task=task, client=self.client, settings=self.settings
                ).run()

        for index, wave in enumerate(waves):
            logger.info("wave_start", wave=index, task_count=len(wave))
            results = await asyncio.gather(
                *[_execute(task) for task in wave], return_exceptions=True
            )
            for task, result in zip(wave, results):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    logger.error("task_failed", task_id=task.id, error=str(result))
                else:
                    task.status = "completed"
                    task.result = result
                    console.print(f"[green]Task {task.id} completed[/green]")
                    logger.info("task_completed", task_id=task.id)

        try:
            with Status("[bold yellow]Testing...", console=console):
                test_result = await TesterAgent(
                    client=self.client, settings=self.settings
                ).run(self.goal, plan.tasks)
        except Exception as e:
            test_result = TestResult(passed=False, summary=str(e), failures=[str(e)])
            console.print(f"[red]Testing failed: {e}[/red]")
            logger.error("testing_failed", error=str(e))

        if test_result.passed:
            self.last_test_passed = True
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("testing_complete", passed=True, summary=test_result.summary)
        else:
            self.last_test_passed = False
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("testing_complete", passed=False, summary=test_result.summary)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _topological_sort(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        task_map = {t.id: t for t in tasks}
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        adjacency: dict[str, list[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dependency in task.dependencies:
                if dependency in task_map:
                    adjacency[dependency].append(task.id)
                    in_degree[task.id] += 1

        waves: list[list[TaskModel]] = []
        current = [tid for tid, deg in in_degree.items() if deg == 0]
        processed: set[str] = set()

        while current:
            wave = [task_map[tid] for tid in current if tid in task_map]
            waves.append(wave)
            processed.update(tid for tid in current if tid in task_map)
            next_wave: list[str] = []
            for tid in current:
                for dependent in adjacency.get(tid, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_wave.append(dependent)
            current = next_wave

        for tid in in_degree:
            if tid not in processed:
                waves.append([task_map[tid]])

        return waves

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        if any(t.status == "failed" for t in tasks):
            return False
        if not self.last_test_passed:
            return False
        return all(t.status == "completed" for t in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.current_tasks
