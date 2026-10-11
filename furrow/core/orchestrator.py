from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

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

# Configure logging
logger = logging.getLogger("furrow.orchestrator")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.state_file = state_file or self.settings.state_file
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _compact_history(self) -> None:
        """Compact task history to prevent context overflow."""
        if len(self.all_tasks) <= 20:
            return
        
        # Keep only last 10 completed tasks and all pending/failed
        completed = [t for t in self.all_tasks if t.status == "completed"]
        others = [t for t in self.all_tasks if t.status != "completed"]
        
        if len(completed) > 10:
            # Summarize older completed tasks
            old_completed = completed[:-10]
            recent_completed = completed[-10:]
            
            summary = f"[Previous {len(old_completed)} tasks completed: " + \
                     ", ".join(f"{t.id}:{t.description[:30]}" for t in old_completed) + "]"
            
            # Add summary as a pseudo-task
            summary_task = TaskModel(
                id="summary",
                description=summary,
                status="completed",
                result="History compacted"
            )
            self.all_tasks = [summary_task] + recent_completed + others
            logger.info(f"Compacted history: {len(old_completed)} tasks summarized")

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        # Load previous state if exists
        await self._load_state()
        
        while True:
            self.cycles += 1
            
            # Check max_cycles limit
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Reached max cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break
                
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            
            # Save state after each cycle
            await self._save_state()
            
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Compact history if needed
        self._compact_history()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal, self.all_tasks)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Filter out tasks that are already completed or have unmet dependencies
        runnable_tasks = self._get_runnable_tasks(plan.tasks)
        
        if not runnable_tasks:
            console.print("[yellow]No runnable tasks (all blocked by dependencies or completed).[/yellow]")
            return

        # Limit parallel tasks
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        
        async def run_with_semaphore(task: TaskModel) -> tuple[TaskModel, str | Exception]:
            async with semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                    return task, result
                except Exception as e:
                    return task, e

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [run_with_semaphore(task) for task in runnable_tasks]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(runnable_tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
            
            # Update task in all_tasks
            self._update_task_in_history(task)

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, self.all_tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _get_runnable_tasks(self, new_tasks: list[TaskModel]) -> list[TaskModel]:
        """Get tasks that can run (dependencies met and not already completed)."""
        runnable = []
        for task in new_tasks:
            # Check if already in history with completed status
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if existing and existing.status == "completed":
                continue
            
            # Check dependencies
            deps_met = True
            for dep_id in task.dependencies:
                dep_task = next((t for t in self.all_tasks if t.id == dep_id), None)
                if not dep_task or dep_task.status != "completed":
                    deps_met = False
                    break
            
            if deps_met:
                runnable.append(task)
            else:
                console.print(f"[yellow]Task {task.id} blocked by unmet dependencies[/yellow]")
        
        return runnable

    def _update_task_in_history(self, task: TaskModel) -> None:
        """Update or add task to history."""
        existing_idx = next((i for i, t in enumerate(self.all_tasks) if t.id == task.id), None)
        if existing_idx is not None:
            self.all_tasks[existing_idx] = task
        else:
            self.all_tasks.append(task)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        
        if failed > 0:
            return False
        if completed >= len(self.all_tasks):
            return True
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks

    async def _save_state(self) -> None:
        """Save orchestrator state to file."""
        state = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [task.model_dump() for task in self.all_tasks],
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            async with asyncio.Lock():
                import aiofiles
                async with aiofiles.open(self.state_file, "w") as f:
                    await f.write(json.dumps(state, indent=2))
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def _load_state(self) -> None:
        """Load orchestrator state from file."""
        if not self.state_file.exists():
            return
        
        try:
            import aiofiles
            async with aiofiles.open(self.state_file, "r") as f:
                content = await f.read()
            state = json.loads(content)
            
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            self.all_tasks = [TaskModel(**t) for t in state.get("tasks", [])]
            
            console.print(f"[cyan]Loaded state: {len(self.all_tasks)} tasks, {self.cycles} cycles[/cyan]")
            logger.info(f"State loaded from {self.state_file}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")