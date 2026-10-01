from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import (
    OrchestratorState,
    Plan,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    get_logger,
)
from furrow.llm import LLMClient

log = get_logger("furrow.orchestrator")


class Orchestrator:
    """Runs the Furrow infinite development loop: plan → execute → test → repeat."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or (client.settings if client else Settings())
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.output_callback = output_callback
        self._console = Console()
        self._callback_console: Console | None = None
        self.state: OrchestratorState | None = None
        self._tasks: list[TaskModel] = []  # tasks from the current cycle
        self._semaphore: asyncio.Semaphore | None = None
        self._load_state()

    def _emit(self, renderable: RenderableType) -> None:
        """Print to console and optionally forward to output callback (e.g. WebSocket)."""
        self._console.print(renderable)
        if self.output_callback is not None:
            if self._callback_console is None:
                self._callback_console = Console(force_terminal=False, no_color=True, width=200)
            try:
                with self._callback_console.capture() as capture:
                    self._callback_console.print(renderable)
                self.output_callback(capture.get())
            except (OSError, ValueError, TypeError) as e:
                log.warning("orchestrator.output_callback_error", error=str(e))

    def _load_state(self) -> None:
        """Load persistent state from the state file if it exists."""
        if not self.settings.state_file:
            return
        state_path = self.settings.workspace / self.settings.state_file
        try:
            if state_path.exists():
                data = json.loads(state_path.read_text())
                self.state = OrchestratorState(**data)
                log.info("orchestrator.state_loaded", goal=self.state.goal, cycle=self.state.cycle)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            log.warning("orchestrator.state_load_failed", error=str(e))
            self.state = None

    def _save_state(self) -> None:
        """Persist orchestrator state to the state file."""
        if not self.settings.state_file:
            return
        if self.state is None:
            self.state = OrchestratorState(goal=self.goal, cycle=self.cycles)
        self.state.cycle = self.cycles
        self.state.goal = self.goal
        # Sync current tasks into history (deduplicating by id)
        for task in self._tasks:
            self.state.add_task(task)
        state_path = self.settings.workspace / self.settings.state_file
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(self.state.model_dump_json(indent=2))
            log.info("orchestrator.state_saved", path=str(state_path))
        except OSError as e:
            log.warning("orchestrator.state_save_failed", error=str(e))

    def _get_tasks(self) -> list[TaskModel]:
        """Return the list of tasks from the current cycle."""
        return self._tasks

    async def run(self) -> None:
        """Run the orchestrator loop until the goal is complete or max_cycles is reached."""
        self._emit(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )

        while True:
            if self._should_stop():
                self._emit(
                    f"[yellow]Stopping: reached max_cycles ({self.settings.max_cycles})[/yellow]"
                )
                break

            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            await self._cycle()

            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                if self.state is not None:
                    self.state.completed = True
                self._save_state()
                break

            # Brief pause between cycles to avoid tight loops
            await asyncio.sleep(0.5)

        await self.client.aclose()

    def _should_stop(self) -> bool:
        """Check if the orchestrator should stop (max_cycles reached).

        max_cycles == 0 means unlimited cycles (the default).
        """
        return self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles

    async def _cycle(self) -> None:
        """Run a single planning → execution → testing cycle."""
        with Status("[bold yellow]Planning...", console=self._console):
            plan: Plan = await self.planner.plan(self.goal)

        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._tasks = []
            self._save_state()
            return

        # Track tasks for this cycle
        self._tasks = plan.tasks
        self._save_state()

        # Execute tasks in parallel, respecting max_parallel_tasks
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> TaskModel:
            async with self._semaphore:
                task.status = TaskStatus.IN_PROGRESS.value
                try:
                    worker = WorkerAgent(task=task, client=self.client)
                    result = await worker.run()
                    task.status = TaskStatus.COMPLETED.value
                    task.result = result
                except Exception as e:
                    task.status = TaskStatus.FAILED.value
                    task.result = str(e)
                    log.error("orchestrator.task_failed", task_id=task.id, error=str(e))
                return task

        with Status("[bold yellow]Executing tasks in parallel...", console=self._console):
            results = await asyncio.gather(*[run_task(t) for t in plan.tasks])

        # Update task list with results
        self._tasks = list(results)
        for task in self._tasks:
            if task.status == "failed":
                self._emit(f"[red]Task {task.id} failed: {task.result}[/red]")
            else:
                self._emit(f"[green]Task {task.id} completed[/green]")

        # Update state with completed tasks
        if self.state is not None:
            for task in self._tasks:
                self.state.update_task(task)

        # Testing phase
        with Status("[bold yellow]Testing...", console=self._console):
            test_result: TestResult = await TesterAgent(client=self.client).run(
                self.goal, self._tasks
            )

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")

        # Record test result in state
        if self.state is not None:
            self.state.test_history.append(test_result)

        # Update goal to focus on fixing failures if any
        if not test_result.passed:
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        self._save_state()

    def _is_done(self) -> bool:
        """Determine if the goal is complete based on task results.

        Goal is considered done when:
        - All tasks in the current cycle are completed (not pending/in_progress/failed)
        """
        tasks = self._get_tasks()
        if not tasks:
            # No tasks means the planner thinks the goal is already done
            return True

        all_completed = all(t.status == TaskStatus.COMPLETED.value for t in tasks)
        all_failed = all(t.status == TaskStatus.FAILED.value for t in tasks)

        if all_failed:
            return False

        return all_completed
