from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import ConsoleCallback, FurrowState, Plan, Settings, TaskModel, TestResult, logger
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, on_output: ConsoleCallback | None = None, state_path: Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.settings = Settings()
        self.cycles = 0
        self.current_tasks: list[TaskModel] = []
        self.on_output = on_output
        self.state_path = state_path or (self.settings.workspace / ".furrow_state.json")
        self.state = FurrowState(goal=goal, cycle=0, tasks=[], completed=[], failed=[])

    def _emit(self, text: str) -> None:
        console.print(text)
        if self.on_output:
            self.on_output(text)

    def _save_state(self) -> None:
        try:
            completed = [t.id for t in self.current_tasks if t.status == "completed"]
            failed = [t.id for t in self.current_tasks if t.status == "failed"]
            self.state = FurrowState(
                goal=self.goal,
                cycle=self.cycles,
                tasks=self.current_tasks,
                completed=completed,
                failed=failed,
                last_plan=None,
            )
            self.state.save(self.state_path)
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("orchestrator_started", goal=self.goal, max_cycles=self.settings.max_cycles)
        if self.state_path.exists():
            try:
                self.state = FurrowState.load(self.state_path)
                self.goal = self.state.goal
                self.cycles = self.state.cycle
                self.current_tasks = self.state.tasks
                logger.info("state_loaded", path=str(self.state_path), cycle=self.cycles)
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Halting.[/yellow]")
                logger.warning("max_cycles_reached", cycles=self.cycles, max_cycles=self.settings.max_cycles)
                break
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        self.current_tasks = plan.tasks
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("plan_created", task_count=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.current_tasks
