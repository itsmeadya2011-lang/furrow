from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, StateManager, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, settings: Settings | None = None, resume: bool = False) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings or self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.state = StateManager(self.settings.state_file)
        self.cycles = 0
        self.resume = resume
        self._callback: Callable[[dict], Awaitable[None]] | None = None
        
        if resume and self.state.current_goal:
            console.print(f"[cyan]Resuming previous session: {self.state.current_goal}[/cyan]")
            self.goal = self.state.current_goal
            self.cycles = len(self.state.cycles)

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        
        max_cycles = self.settings.max_cycles or 100
        
        while self.cycles < max_cycles:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            
            if self.state.is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
                
            # Check for user interrupt
            await asyncio.sleep(0.1)

    async def run_with_callback(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._callback = callback
        await self._send({"type": "log", "message": f"Starting Furrow with goal: {self.goal}"})
        
        max_cycles = self.settings.max_cycles or 100
        
        while self.cycles < max_cycles:
            self.cycles += 1
            await self._send({"type": "cycle_start", "cycle": self.cycles})
            await self._cycle()
            
            if self.state.is_done():
                await self._send({"type": "done", "message": "Goal complete. Halting."})
                break
                
            await asyncio.sleep(0.1)

    async def _send(self, msg: dict) -> None:
        if self._callback:
            await self._callback(msg)

    async def _cycle(self) -> None:
        # Get pending tasks from state if resuming
        if self.resume and self.cycles <= len(self.state.cycles):
            # We're resuming, use existing plan
            pending_tasks = self.state.get_pending_tasks()
            if pending_tasks:
                plan = Plan(tasks=pending_tasks, rationale="Resumed from previous session")
            else:
                plan = await self._plan()
        else:
            plan = await self._plan()

        if not plan.tasks:
            await self._send({"type": "log", "message": "No tasks planned. Goal may be complete."})
            return

        # Update task status to in_progress
        for task in plan.tasks:
            task.status = "in_progress"
            task.cycle = self.cycles

        await self._send({
            "type": "plan",
            "rationale": plan.rationale,
            "tasks": [{"id": t.id, "description": t.description, "files": t.files} for t in plan.tasks]
        })

        # Execute tasks in parallel
        tasks = [
            WorkerAgent(task=task, client=self.client, settings=self.settings).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            task.attempts += 1
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._send({"type": "task_complete", "task_id": task.id, "status": "failed", "result": str(result)})
            else:
                task.status = "completed"
                task.result = result
                await self._send({"type": "task_complete", "task_id": task.id, "status": "completed", "result": result})

        # Run tests
        test_result = await TesterAgent(client=self.client, settings=self.settings).run(self.goal, plan.tasks)

        await self._send({
            "type": "test_result",
            "passed": test_result.passed,
            "summary": test_result.summary,
            "failures": test_result.failures,
        })

        # Save state
        self.state.add_cycle(self.cycles, self.goal, plan.tasks, test_result)
        
        # Print summary to console as well
        self._print_summary(plan.tasks, test_result)

    async def _plan(self) -> Plan:
        # Get project context for planner
        files = self.client.list_files(self.settings.workspace)
        context = f"Project files: {', '.join(files[:50])}"
        if len(files) > 50:
            context += f" ... and {len(files) - 50} more"
        
        # Include completed task summaries
        completed = self.state.get_completed_tasks()
        if completed:
            context += "\n\nCompleted tasks:\n"
            for t in completed[-5:]:
                context += f"  - {t.id}: {t.description} -> {t.result[:100]}\n"
        
        plan = await self.planner.plan(self.goal, context)
        return plan

    def _print_summary(self, tasks: list[TaskModel], test_result: TestResult) -> None:
        table = Table(title=f"Cycle {self.cycles} Summary")
        table.add_column("Task ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Result", style="dim")
        
        for task in tasks:
            status_style = {
                "completed": "[green]✓[/green]",
                "failed": "[red]✗[/red]",
                "in_progress": "[yellow]⟳[/yellow]",
                "pending": "[dim]○[/dim]",
            }.get(task.status, task.status)
            table.add_row(task.id, task.description[:50], status_style, (task.result or "")[:60])
        
        console.print(table)
        console.print(f"Tests: {'[green]PASSED[/green]' if test_result.passed else '[red]FAILED[/red]'} - {test_result.summary}")
