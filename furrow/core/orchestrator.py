from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from typing import Any

import aiofiles
import structlog
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
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.plan: Plan | None = None
        self.state_file = Path(state_file) if state_file else Path.cwd() / ".furrow_state.json"
        self._shutdown = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (ValueError, OSError):
                pass  # Not in main thread, ignore

    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info("Received shutdown signal", signal=signum)
        self._shutdown = True
        console.print("\n[yellow]Shutdown requested. Finishing current cycle...[/yellow]")

    async def run(self) -> None:
        """Run the orchestrator loop until goal is complete or max cycles reached."""
        # Load previous state if exists
        await self._load_state()

        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))

        while not self._shutdown:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print(f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]")
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("Starting cycle", cycle=self.cycles, goal=self.goal)

            await self._cycle()
            await self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("Goal completed", cycles=self.cycles)
                break

        # Clean up state file on successful completion
        if self._is_done() and self.state_file.exists():
            self.state_file.unlink()

    async def _cycle(self) -> None:
        """Execute one cycle: plan -> execute -> test."""
        # Planning phase
        with Status("[bold yellow]Planning...", console=console):
            self.plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(self.plan.model_dump()), title="Plan", border_style="blue"))
        logger.info("Plan created", tasks=len(self.plan.tasks), rationale=self.plan.rationale)

        if not self.plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Execution phase - respect dependencies
        await self._execute_tasks()

        # Testing phase
        with Status("[bold yellow]Testing...", console=console):
            test_result = await self.tester.run(self.goal, self.plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("Tests passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("Tests failed", failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self) -> None:
        """Execute tasks respecting dependencies and max parallelism."""
        if not self.plan:
            return

        # Build dependency graph
        task_map = {task.id: task for task in self.plan.tasks}
        completed: set[str] = set()
        failed: set[str] = set()

        # Group tasks by dependency level
        remaining = set(task_map.keys())

        while remaining and not self._shutdown:
            # Find tasks that can run (dependencies met)
            runnable = [
                task_id for task_id in remaining
                if all(dep in completed for dep in task_map[task_id].dependencies)
            ]

            if not runnable:
                # Circular dependency or all remaining tasks blocked
                blocked = remaining - completed - failed
                if blocked:
                    console.print(f"[red]Blocked tasks (circular deps?): {', '.join(blocked)}[/red]")
                    logger.error("Tasks blocked by unmet dependencies", blocked=list(blocked))
                    for task_id in blocked:
                        task_map[task_id].status = "failed"
                        task_map[task_id].result = "Blocked by unmet dependencies"
                break

            # Limit parallelism
            batch = runnable[: self.settings.max_parallel_tasks]
            remaining -= set(batch)

            console.print(f"[cyan]Executing {len(batch)} task(s) in parallel...[/cyan]")

            tasks = [WorkerAgent(task=task_map[tid], client=self.client, settings=self.settings).run() for tid in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for task_id, result in zip(batch, results):
                task = task_map[task_id]
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    console.print(f"[red]Task {task_id} failed: {result}[/red]")
                    logger.error("Task failed", task_id=task_id, error=str(result))
                    failed.add(task_id)
                else:
                    task.status = "completed"
                    task.result = result  # type: ignore[assignment]
                    console.print(f"[green]Task {task_id} completed[/green]")
                    logger.info("Task completed", task_id=task_id)
                    completed.add(task_id)

            # Small delay between batches
            if remaining:
                await asyncio.sleep(0.5)

        # Retry failed tasks once
        if failed and not self._shutdown:
            console.print(f"[yellow]Retrying {len(failed)} failed task(s)...[/yellow]")
            await self._retry_failed_tasks(task_map, failed)

    async def _retry_failed_tasks(self, task_map: dict[str, TaskModel], failed_ids: set[str]) -> None:
        """Retry failed tasks once with updated context."""
        for task_id in list(failed_ids):
            if self._shutdown:
                break
            task = task_map[task_id]
            console.print(f"[yellow]Retrying task {task_id}...[/yellow]")
            task.status = "pending"  # Reset for retry
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            result = await worker.run()
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = f"Retry failed: {result}"
                console.print(f"[red]Task {task_id} failed on retry: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task_id} completed on retry[/green]")

    def _is_done(self) -> bool:
        """Check if all tasks are completed successfully."""
        if not self.plan or not self.plan.tasks:
            return True

        completed = sum(1 for t in self.plan.tasks if t.status == "completed")
        failed = sum(1 for t in self.plan.tasks if t.status == "failed")

        if failed > 0:
            return False
        return completed >= len(self.plan.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        """Return current plan tasks."""
        return self.plan.tasks if self.plan else []

    async def _save_state(self) -> None:
        """Persist orchestrator state to file."""
        if not self.plan:
            return

        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "plan": self.plan.model_dump(),
        }
        try:
            async with aiofiles.open(self.state_file, "w") as f:
                await f.write(json.dumps(state, indent=2))
            logger.debug("State saved", file=str(self.state_file))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to save state", error=str(e))

    async def _load_state(self) -> None:
        """Load orchestrator state from file if exists."""
        if not self.state_file.exists():
            return

        try:
            async with aiofiles.open(self.state_file, "r") as f:
                content = await f.read()
            state = json.loads(content)

            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            if "plan" in state:
                self.plan = Plan(**state["plan"])

            console.print(f"[cyan]Resumed from cycle {self.cycles}[/cyan]")
            logger.info("State loaded", cycles=self.cycles, goal=self.goal)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load state, starting fresh", error=str(e))

    def get_status(self) -> dict[str, Any]:
        """Get current orchestrator status for web UI."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "plan": self.plan.model_dump() if self.plan else None,
            "tasks": [t.model_dump() for t in self._get_tasks()],
            "done": self._is_done(),
        }