from __future__ import annotations

import asyncio
import json
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TestResult, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Coordinates planning, execution, and testing cycles toward a goal."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        no_test: bool = False,
        workspace: Path | None = None,
        on_output=None,
        stop_event=None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.no_test = no_test
        self.tasks: list[TaskModel] = []
        ws = workspace or self.settings.workspace
        self.state_path = Path(ws) / (goal.replace(" ", "_")[:50] + "_state.json")
        self.logger = structlog.get_logger()
        self._on_output = on_output
        self._stop_event = stop_event

    def _output(self, message: str) -> None:
        if self._on_output:
            self._on_output(message)
        console.print(message)

    async def run(self) -> None:
        """Run the orchestration loop until the goal is complete or cycles are exhausted."""
        self._output(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        self.load_state()
        while True:
            self.cycles += 1
            self.logger.info("cycle_start", cycle=self.cycles, goal=self.goal)
            self._output(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self.logger.info("goal_complete", cycles=self.cycles)
                self._output("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                self.logger.info("max_cycles_reached", max_cycles=self.max_cycles)
                self._output(f"[yellow]Max cycles ({self.max_cycles}) reached. Halting.[/yellow]")
                break
            if self._stop_event and self._stop_event.is_set():
                self.logger.info("stopped_by_user")
                self._output("[yellow]Stopped by user.[/yellow]")
                break

    async def _cycle(self) -> None:
        """Execute a single planning -> execution -> testing cycle."""
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._output(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self.logger.info("plan", tasks=[t.model_dump() for t in plan.tasks], rationale=plan.rationale)

        self.tasks = plan.tasks

        if not self.tasks:
            self.logger.info("no_tasks_planned")
            self._output("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute_tasks()

        with Status("[bold yellow]Testing...", console=console):
            if self.no_test:
                test_result = TestResult(passed=True, summary="Testing skipped", failures=[])
            else:
                test_result = await TesterAgent(client=self.client).run(self.goal, self.tasks)

        await self._handle_test_result(test_result)
        self.save_state()

    async def _execute_tasks(self) -> None:
        """Run all pending tasks in parallel and update their statuses."""
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [WorkerAgent(task=task, client=self.client).run() for task in self.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.logger.error("task_failed", task_id=task.id, error=str(result))
                self._output(f"[red]Task {task.id} failed: {result}[/red]")
                await self._spawn_fix_worker(task, str(result))
            else:
                task.status = "completed"
                task.result = json.dumps(result) if isinstance(result, dict) else str(result)
                self.logger.info("task_completed", task_id=task.id, result=result)
                self._output(f"[green]Task {task.id} completed[/green]")

    async def _spawn_fix_worker(self, task: TaskModel, error: str) -> None:
        """Spawn a fix worker immediately in the same cycle for a failed task."""
        fix_goal = f"Fix failing task {task.id}:\nTask: {task.description}\nError: {error}"
        self.logger.info("spawning_fix_worker", task_id=task.id)
        self._output(f"[yellow]Spawning fix worker for {task.id}...[/yellow]")
        with Status("[bold yellow]Fixing...", console=console):
            try:
                result = await WorkerAgent(task=task, client=self.client).run()
                task.status = "completed"
                task.result = json.dumps(result) if isinstance(result, dict) else str(result)
                self.logger.info("fix_worker_succeeded", task_id=task.id)
                self._output(f"[green]Task {task.id} fixed[/green]")
            except Exception as exc:
                self.logger.error("fix_worker_failed", task_id=task.id, error=str(exc))
                self._output(f"[red]Fix worker for {task.id} failed: {exc}[/red]")

    async def _handle_test_result(self, test_result: TestResult) -> None:
        """Apply test outcomes: clear completed tasks or update goal for fixes."""
        if test_result.passed:
            self.logger.info("tests_passed", summary=test_result.summary)
            self._output(f"[green]Tests passed: {test_result.summary}[/green]")
            self.tasks = [t for t in self.tasks if t.status != "completed"]
        else:
            self.logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self._output(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._output(f"  • {failure}")
            self._output("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        """Determine whether the orchestration loop should halt."""
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        if failed > 0:
            return False
        if tasks and completed >= len(tasks):
            return True
        if not tasks:
            self.logger.info("no_tasks_remaining")
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        """Return the current task list."""
        return self.tasks

    def save_state(self) -> None:
        """Persist orchestrator state to disk for resumption."""
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }
        try:
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=2)
            self.logger.info("state_saved", path=str(self.state_path))
        except OSError as exc:
            self.logger.error("state_save_failed", path=str(self.state_path), error=str(exc))

    def load_state(self) -> None:
        """Load orchestrator state from disk if it exists."""
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            self.tasks = [TaskModel(**t) for t in state.get("tasks", [])]
            self.logger.info("state_loaded", path=str(self.state_path), cycles=self.cycles)
        except (OSError, ValueError) as exc:
            self.logger.error("state_load_failed", path=str(self.state_path), error=str(exc))