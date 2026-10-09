from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult, settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient(settings=config or settings)
        self.planner = PlannerAgent(client=self.client, settings=config or settings)
        self.tester = TesterAgent(client=self.client, settings=config or settings)
        self.config = config or settings
        self.cycles = 0
        self.tasks: dict[str, TaskModel] = {}
        self.state_file = Path(self.config.workspace) / self.config.state_file
        self.start_time = time.time()
        self._load_state()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                for task_data in data.get("tasks", []):
                    task = TaskModel(**task_data)
                    self.tasks[task.id] = task
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.tasks))
            except Exception as e:  # noqa: BLE001
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self.tasks.values()],
            "start_time": self.start_time,
        }
        self.state_file.write_text(json.dumps(data, indent=2))
        logger.debug("state_saved", tasks=len(self.tasks))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal)

        try:
            while True:
                self.cycles += 1
                cycle_start = time.time()
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                logger.info("cycle_started", cycle=self.cycles)

                await self._cycle()

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info("orchestrator_completed", cycles=self.cycles)
                    break

                if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                    console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Stopping.[/yellow]")
                    logger.info("max_cycles_reached", cycles=self.cycles)
                    break

                cycle_duration = time.time() - cycle_start
                logger.info("cycle_completed", cycle=self.cycles, duration=cycle_duration)

        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            logger.info("orchestrator_interrupted", cycles=self.cycles)
        except Exception as e:
            logger.exception("orchestrator_error", error=str(e))
            raise
        finally:
            self._save_state()

    async def _cycle(self) -> None:
        plan = await self._plan_with_retry()

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        await self._execute_tasks(plan)

        test_result = await self._test_with_retry(plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._save_state()
        self._print_status()

    async def _plan_with_retry(self) -> Plan:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with Status("[bold yellow]Planning...", console=console):
                    plan = await self.planner.plan(self.goal, self.tasks)
                return plan
            except Exception as e:
                logger.warning("plan_failed", attempt=attempt + 1, error=str(e))
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        raise RuntimeError("Failed to create plan after retries")

    async def _execute_tasks(self, plan: Plan) -> None:
        sorted_tasks = self._topological_sort(plan.tasks)

        semaphore = asyncio.Semaphore(self.config.max_parallel_tasks)

        async def run_task(task: TaskModel, sem: asyncio.Semaphore = semaphore) -> str | BaseException:
            async with sem:
                return await self._run_single_task(task)

        task_results = await asyncio.gather(
            *[run_task(task) for task in sorted_tasks],
            return_exceptions=True,
        )

        for task, result in zip(sorted_tasks, task_results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                task.retry_count += 1
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = str(result) if isinstance(result, BaseException) else result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)
            self.tasks[task.id] = task

    async def _run_single_task(self, task: TaskModel) -> str:
        if task.status == "completed":
            return task.result or "Already completed"

        missing_deps = [dep for dep in task.dependencies if self.tasks.get(dep, TaskModel(id=dep, description="")).status != "completed"]
        if missing_deps:
            logger.info("task_waiting_for_deps", task_id=task.id, deps=missing_deps)
            return f"Waiting for dependencies: {', '.join(missing_deps)}"

        worker = WorkerAgent(task=task, client=self.client, settings=self.config)
        return await asyncio.wait_for(worker.run(), timeout=self.config.task_timeout)

    def _topological_sort(self, tasks: list[TaskModel]) -> list[TaskModel]:
        task_map = {t.id: t for t in tasks}
        visited = set()
        temp = set()
        result = []

        def visit(task_id: str) -> None:
            if task_id in temp:
                raise ValueError(f"Circular dependency detected: {task_id}")
            if task_id in visited:
                return
            temp.add(task_id)
            task = task_map.get(task_id)
            if task:
                for dep in task.dependencies:
                    if dep in task_map:
                        visit(dep)
            temp.remove(task_id)
            visited.add(task_id)
            if task:
                result.append(task)

        for task in tasks:
            visit(task.id)

        return result

    async def _test_with_retry(self, tasks: list[TaskModel]) -> TestResult:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                with Status("[bold yellow]Testing...", console=console):
                    test_result = await self.tester.run(self.goal, tasks)
                return test_result
            except Exception as e:  # noqa: BLE001
                logger.warning("test_failed", attempt=attempt + 1, error=str(e))
                if attempt == max_retries - 1:
                    return TestResult(passed=False, summary=str(e), failures=[str(e)])
                await asyncio.sleep(2**attempt)
        return TestResult(passed=False, summary="Test retries exhausted", failures=["Test retries exhausted"])

    def _is_done(self) -> bool:
        pending = [t for t in self.tasks.values() if t.status == "pending"]
        in_progress = [t for t in self.tasks.values() if t.status == "in_progress"]
        failed = [t for t in self.tasks.values() if t.status == "failed"]

        if failed and all(t.retry_count >= t.max_retries for t in failed):
            console.print("[red]Max retries exceeded for failed tasks.[/red]")
            return True

        if failed and any(t.retry_count < t.max_retries for t in failed):
            return False

        return bool(not pending and not in_progress)

    def _print_status(self) -> None:
        table = Table(title="Task Status")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Retries", style="yellow")
        table.add_column("Dependencies", style="dim")

        for task in sorted(self.tasks.values(), key=lambda t: t.id):
            status_style = {
                "pending": "yellow",
                "in_progress": "blue",
                "completed": "green",
                "failed": "red",
            }.get(task.status, "white")

            table.add_row(
                task.id,
                task.description[:60] + ("..." if len(task.description) > 60 else ""),
                f"[{status_style}]{task.status}[/{status_style}]",
                str(task.retry_count) if task.retry_count > 0 else "",
                ", ".join(task.dependencies) if task.dependencies else "—",
            )

        console.print(table)

        elapsed = time.time() - self.start_time
        console.print(f"[dim]Elapsed: {elapsed:.1f}s | Cycles: {self.cycles}[/dim]")