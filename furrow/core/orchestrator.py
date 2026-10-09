from __future__ import annotations

import asyncio
import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

# Configure logging
configure_logging(settings.log_level, settings.log_file)
log = get_logger("furrow.orchestrator")

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, config: Settings | None = None) -> None:
        self.goal = goal
        self.config = config or settings
        self.client = client or LLMClient(settings=self.config)
        self.planner = PlannerAgent(client=self.client, settings=self.config)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = self.config.workspace / ".furrow_state.json"

    async def run(self) -> None:
        await self._load_state()
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        log.info("orchestrator_started", goal=self.goal, workspace=str(self.config.workspace))

        while True:
            if self.config.max_cycles > 0 and self.cycles >= self.config.max_cycles:
                console.print(f"[yellow]Max cycles ({self.config.max_cycles}) reached. Stopping.[/yellow]")
                log.warning("max_cycles_reached", max_cycles=self.config.max_cycles)
                break

            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            log.info("cycle_started", cycle=self.cycles)
            await self._cycle()
            await self._save_state()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info(
                    "orchestrator_completed",
                    cycles=self.cycles,
                    tasks_completed=len([t for t in self.all_tasks if t.status == "completed"]),
                )
                break

    async def _cycle(self) -> None:
        # Filter pending tasks from previous cycles
        pending_tasks = [t for t in self.all_tasks if t.status == "pending"]

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, pending_tasks)

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        log.info("plan_created", cycle=self.cycles, num_tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No new tasks planned. Goal may be complete.[/yellow]")
            log.info("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our task list
        for task in plan.tasks:
            task.status = "pending"
            self.all_tasks.append(task)

        # Get all pending tasks (including newly added)
        tasks_to_run = [t for t in self.all_tasks if t.status == "pending"]

        # Limit parallel tasks
        tasks_to_run = tasks_to_run[: self.config.max_parallel_tasks]

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.config).run()
                for task in tasks_to_run
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                log.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                log.info(
                    "task_completed",
                    task_id=task.id,
                    summary=result[:100] if isinstance(result, str) else str(result)[:100],
                )

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.config).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            log.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            log.warning("tests_failed", cycle=self.cycles, summary=test_result.summary, failures=test_result.failures)
            # Update goal to focus on fixing failures
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")

        # If there are failures, not done
        if failed > 0:
            return False

        # If no pending tasks and all completed, we're done
        return bool(pending == 0 and completed > 0)

    async def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                content = await self.client.read_file(self.state_file)
                data = json.loads(content)
                self.goal = data.get("goal", self.goal)
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                console.print(f"[dim]Loaded state: {len(self.all_tasks)} tasks, cycle {self.cycles}[/dim]")
                log.info("state_loaded", tasks=len(self.all_tasks), cycle=self.cycles)
            except Exception as e:  # noqa: BLE001
                console.print(f"[yellow]Could not load state: {e}[/yellow]")
                log.warning("state_load_failed", error=str(e))

    async def _save_state(self) -> None:
        try:
            data = {
                "goal": self.goal,
                "cycles": self.cycles,
                "tasks": [t.model_dump() for t in self.all_tasks],
                "updated": datetime.now(datetime.timezone.utc).isoformat(),
            }
            content = json.dumps(data, indent=2)
            await self.client.write_file(self.state_file, content)
            log.debug("state_saved", tasks=len(self.all_tasks), cycle=self.cycles)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
            log.warning("state_save_failed", error=str(e))