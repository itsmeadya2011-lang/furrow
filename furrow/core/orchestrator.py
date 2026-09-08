from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable

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
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

# Type for the optional event callback: (event_type, message, css_class) -> None
EventCallback = Callable[[str, str, str], Awaitable[None]] | None


class Orchestrator:
    """Central loop controller for the Furrow autonomous development loop.

    The orchestrator runs cycles of: Plan → Execute (parallel workers) → Test → Repeat.
    It tracks task state across cycles and respects the ``max_cycles`` setting.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
        on_event: EventCallback = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: dict[str, TaskModel] = {}
        self.test_results: list[TestResult] = []
        self._workspace = workspace or self.settings.workspace
        self._state_file = self._workspace / ".furrow-state.json"
        self._on_event = on_event

    @property
    def workspace(self) -> Path:
        return self._workspace

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tasks tracked across cycles."""
        return list(self.tasks.values())

    def _register_tasks(self, plan: Plan) -> None:
        """Register tasks from a plan into the orchestrator's task registry."""
        for task in plan.tasks:
            self.tasks[task.id] = task

    def _is_done(self) -> bool:
        """Check if the goal is fully complete.

        Returns True if all tracked tasks are completed and none have failed,
        or if no tasks exist (plan is empty, goal is complete).
        """
        all_tasks = self._get_tasks()
        if not all_tasks:
            return True
        completed = sum(1 for t in all_tasks if t.status == "completed")
        failed = sum(1 for t in all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(all_tasks)

    def _max_cycles_reached(self) -> bool:
        """Check if the maximum number of cycles has been reached."""
        if self.settings.max_cycles <= 0:
            return False
        return self.cycles >= self.settings.max_cycles

    async def _emit(self, event_type: str, message: str, cls: str = "") -> None:
        """Emit an event via the callback and log it."""
        logger.info("Event", event_type=event_type, message=message)
        if self._on_event is not None:
            await self._on_event(event_type, message, cls)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        """Load persisted state from disk if available."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text())
                for task_data in data.get("tasks", []):
                    task = TaskModel(**task_data)
                    self.tasks[task.id] = task
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                logger.info("Loaded state from disk", path=str(self._state_file))
        except Exception as e:
            logger.warning("Failed to load state", error=str(e))

    def _save_state(self) -> None:
        """Persist current state to disk."""
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self._get_tasks()],
            }
            self._workspace.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save state", error=str(e))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the infinite development loop until done or stopped."""
        self._load_state()
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        logger.info("Starting Furrow loop", goal=self.goal, max_cycles=self.settings.max_cycles)
        await self._emit("status", f"Started Furrow: {self.goal}")

        try:
            while True:
                if self._max_cycles_reached():
                    console.print(
                        f"[yellow]Max cycles ({self.settings.max_cycles}) reached. "
                        f"Halting.[/yellow]"
                    )
                    logger.info("Max cycles reached, halting")
                    await self._emit("event", f"Max cycles ({self.settings.max_cycles}) reached. Halting.", "task-fail")
                    break

                self.cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                logger.info("Starting cycle", cycle=self.cycles)
                await self._emit("cycle", f"Cycle {self.cycles}", "cycle")

                await self._cycle()
                self._save_state()

                if self._is_done():
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    logger.info("Goal complete, halting")
                    await self._emit("done", "Goal complete. Halting.")
                    break
        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            logger.info("Stopped by user (KeyboardInterrupt)")
            await self._emit("event", "Stopped by user", "task-fail")
        finally:
            await self.client.close()
            # Clean up state file on clean exit
            if self._is_done():
                try:
                    self._state_file.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _cycle(self) -> None:
        """Execute a single plan → execute → test cycle."""
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("Plan generated", tasks=len(plan.tasks), rationale=plan.rationale)
        await self._emit("status", f"Plan generated: {len(plan.tasks)} tasks", "cycle")

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Register tasks into the tracked registry
        self._register_tasks(plan)

        # Execute tasks in parallel (respecting max_parallel_tasks)
        max_parallel = min(self.settings.max_parallel_tasks, len(plan.tasks))
        sem = asyncio.Semaphore(max_parallel)

        async def _run_with_sem(task: TaskModel) -> str | Exception:
            async with sem:
                return await WorkerAgent(task=task, client=self.client, settings=self.settings).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *[_run_with_sem(task) for task in plan.tasks],
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.tasks[task.id] = task
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("Task failed", task_id=task.id, error=str(result))
                await self._emit("task-fail", f"Task {task.id} failed: {result}")
            else:
                task.status = "completed"
                task.result = result
                self.tasks[task.id] = task
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("Task completed", task_id=task.id)
                await self._emit("task-ok", f"Task {task.id} completed")

        # Test phase
        test_result = await self._run_tests(plan.tasks)
        self.test_results.append(test_result)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed", summary=test_result.summary)
            await self._emit("test-ok", f"Tests passed: {test_result.summary}")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("Tests failed", failures=test_result.failures)
            await self._emit("test-fail", f"Tests failed: {test_result.summary}")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_tests(self, tasks: list[TaskModel]) -> TestResult:
        """Run the tester agent."""
        try:
            tester = TesterAgent(client=self.client, settings=self.settings)
            return await tester.run(self.goal, tasks)
        except Exception as e:
            logger.error("Tester failed", error=str(e))
            return TestResult(passed=False, summary=f"Tester error: {e}", failures=[str(e)])
